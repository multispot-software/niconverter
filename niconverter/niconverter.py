"""
niconverter - Create 48-spot Photon-HDF5 files from NI-FPGA data files.

This modules allows to process 96-ch data files acquired with the NI-FPGA
board and to create Photon-HDF5 files. Since the processing is done in chunks,
arbitrary large data files can be converted using a constant amount of memory.

The input data file must have overflows marked by timestamps equal
to 0 (0x000000). A 0-timestamp can be a "real" count or can be injected
by the FPGA to mark an overflow. In the latter case the detector field
has the highest bit set to 1.

There are two routes to create a Photon-HDF5 file as described next.

- **Route 1**: This single-step approach uses :func:`ni96ch_process_spots`
  which directly saves per-spot timestamps and detectors arrays with
  overflow correction. Photon-HDF5 files can be created by adding metadata
  to the same HDF5 file (callin g :func:`populate_metadata_smFRET_48spots`
  and then :func:`phconvert.save_photon_hdf5`). This approach
  does not create temporary files. However, the implementation is quite
  complex and hard to test directly. The only reason I trust it is because
  this approach was heavily tested using *Route 2*.

- **Route 2**: This approach involves correcting the overflow and saving
  separate timestamps arrays for each channel (not spot!) into a temporary
  file. This steps is done calling :func:`ni96ch_process`.
  Next, we merge the two timestamps arrays for each spot and create the
  detectors arrays. This step is done calling
  :func:`save_timestamps_detectors_48ch`. The alternative function
  :func:`ni96ch_process_inram` save the separate timestamps arrays in RAM,
  and can be used to test the chunking logic. With this route we can
  use two different implementations of the overflow correction. The first,
  separates the channels first and then corrects overflows using numpy.
  The second, corrects the overflow without separating the channels and
  uses numba to speedup the computation.
  This allows splitting the conversion in steps that can be checked
  individually.

"""

from pathlib import Path
import numpy as np
import numba
import tables
from tqdm import tqdm_notebook, tqdm


def iter_chunksize(num_samples, chunksize):
    """Yield `chunksize` enough times to cover `num_samples`.
    """
    for i in range(int(np.ceil(num_samples / chunksize))):
        yield chunksize


def iter_chunk_slice(num_samples, chunksize):
    """Yield slices for iterating over an array in chunks.

    Each iteration returns a slice of size `chunksize`. The itrataion terminates
    when the last slice includes the index (num_sample - 1).
    """
    i = 0
    for c_size in iter_chunksize(num_samples, chunksize):
        yield slice(i, i + c_size)
        i += c_size


def _read_header_NI_LV(fname):
    """Read header of a DAT file saved by MultiCounterProject
    """
    with open(fname, 'rb') as f:
        lines = [f.readline() for _ in range(3)]
        offset = f.tell()
    meta = {'offset': offset}
    for s in lines:
        line = s.decode()
        if 'channels' in line:
            meta['nchannels'] = int(line.split()[-1])
        elif 'words per photon' in line:
            meta['words_per_photon'] = int(line.split()[-1])
        elif 'clock frequency' in line:
            meta['clock_frequency'] = float(line.split()[-1]) * 1e6
    return meta


def detectformat(fname):
    """Detect file type from the header and return format specifications.
    """
    with open(fname, 'rb') as f:
        head = f.read(64)
    format = 'x' if b'channel' in head else 'ni'

    if format == 'x':   # format saved by MultiCounterProject
        meta = _read_header_NI_LV(fname)
        dt = np.dtype([('det', 'u1'), ('time', '3u1')])
        endianess = '>'
    else:               # format saved by FPGA_96ch
        meta = {'nchannels': 96, 'clock_frequency': 80e6, 'offset': 0}
        dt = np.dtype([('time', '3u1'), ('det', 'u1')])
        endianess = '<'
    meta['format'] = format
    with open(fname, 'rb') as f:
        num_timestamps = (f.seek(0, 2) - meta['offset']) / 4
        assert num_timestamps == int(num_timestamps)
        meta['num_timestamps'] = int(num_timestamps)
    return dt, endianess, meta


def duration(ts_m, ts_unit):
    """Compute measurement duration from list of corrected timestamps `ts_m`.
    """
    if isinstance(ts_m[0], list):
        ts_min_l, ts_max_l = [], []
        for t in ts_m:
            if len(t[0]) > 0:
                ts_min_l.append(t[0][0])
                ts_max_l.append(t[-1][-1])
            ts_min, ts_max = min(ts_min_l), max(ts_max_l)
    else:
        ts_min = np.min([t[0] for t in ts_m if len(t) > 0])
        ts_max = np.max([t[-1] for t in ts_m if len(t) > 0])
    return (ts_max - ts_min) * ts_unit


def _inner_loop1(detectors, timestamps, t_start, overflow, nch):
    """Separate channels and apply overflow correction."""
    timestamps_ch = []
    overflow_marker = 0
    for ch in range(nch):
        mask_ch_all = np.bitwise_and(detectors, 127) == ch
        mask_ch_ov = timestamps[mask_ch_all] == overflow_marker
        mask_ch = detectors[mask_ch_all] == ch
        if mask_ch.sum() == 0:
            timestamps_ch.append(np.array([], dtype='int64'))
            continue
        times64 = np.cumsum(mask_ch_ov, dtype='int64') * overflow + t_start[ch]
        t_start[ch] = times64[-1]
        times64 = times64[mask_ch]  # remove overflow timestamps
        times64 += timestamps[mask_ch_all][mask_ch]
        timestamps_ch.append(times64)
    return timestamps_ch


@numba.jit('int64[:](uint8[:], int64[:], int64[:], int64)', nopython=True, nogil=True)
def _overflow_correct_allch(detectors, timestamps, t_start, overflow):
    """Apply overflow correction to all channels."""
    overflow_corr = t_start
    overflow_marker = 0
    for i in range(timestamps.size):
        is_overflow = timestamps[i] == overflow_marker
        det = detectors[i] & 127
        if is_overflow:
            overflow_corr[det] += overflow
        timestamps[i] += overflow_corr[det]
    return timestamps


def _inner_loop2(detectors, timestamps, t_start, overflow, nch):
    """Separate channels and apply overflow correction."""
    new_ts = _overflow_correct_allch(detectors, timestamps, t_start, overflow)
    return [new_ts[detectors == ch] for ch in range(nch)]


def _inner_loop_spots(detectors, timestamps, t_start, overflow):
    """Apply overflow correction and create per-spot timestamps/detectors."""
    new_ts = _overflow_correct_allch(detectors, timestamps, t_start, overflow)
    spots, D_ch, A_ch = get_spot_ch_map_48spots()
    assert (spots == np.arange(48)).all()
    ts_list, det_list = [], []
    for ch_d, ch_a in zip(D_ch, A_ch):
        spot_mask = (detectors == ch_d) + (detectors == ch_a)
        ts_chunk = new_ts[spot_mask]
        det_chunk = detectors[spot_mask]
        index_sorted = ts_chunk.argsort(kind='mergesort')
        ts_list.append(ts_chunk[index_sorted])
        det_list.append(det_chunk[index_sorted] == ch_a)
    return ts_list, det_list


def read_raw_timestamps(fname, num_timestamps=-1):
    """Read `num_timestamps` timestamps and detectors from `fname`.
    """
    nbits = 24
    dt, endianess, meta = detectformat(fname)
    if num_timestamps < meta['num_timestamps']:
        num_timestamps = meta['num_timestamps']
    f = open(fname, 'rb')
    f.seek(meta['offset'])
    timestamps, det = _read_chunk(f, num_timestamps, dt, endianess, nbits)
    return timestamps, det, meta


def _read_chunk(f, chunksize, dt, endianess, nbits):
    """Read `chunksize` bytes from `f` and return raw timestamps and detectors.
    """
    buf = f.read(chunksize * 4)
    det = np.frombuffer(buf, dtype=dt)['det'].copy()
    timestamps = np.frombuffer(buf, dtype='%su4' % endianess)
    timestamps.setflags(write=True)
    np.bitwise_and(timestamps, 2**nbits - 1, out=timestamps)
    timestamps = timestamps.astype(np.int64)
    return timestamps, det


def ni96ch_process_inram(fname, chunksize=2**18, num_timestamps=-1, debug=False,
                         inner_loop=None, progrbar_widget=True):
    """Sort timestamps per-ch and correct overflow in NI-96ch data.
    """
    if inner_loop is None:
        inner_loop = _inner_loop2
    dt, endianess, meta = detectformat(fname)
    if num_timestamps < meta['num_timestamps']:
        num_timestamps = meta['num_timestamps']
    nbits = 24
    overflow = 2**nbits
    nch = meta['nchannels']
    ts_unit = 1 / meta['clock_frequency']

    # Open file and position cursor after header
    f = open(fname, 'rb')
    f.seek(meta['offset'])

    # Separate channels and correct overflow
    timestamps_m = [[] for ch in range(nch)]  # list of timestamps
    t_start = np.zeros(nch, dtype='int64')
    progressbar = tqdm_notebook if progrbar_widget else tqdm
    _iter = progressbar(iter_chunksize(num_timestamps, chunksize),
                        total=np.ceil(num_timestamps / chunksize))
    for chunksize in _iter:
        timestamps, det = _read_chunk(f, chunksize, dt, endianess, nbits)

        ts_chunks = inner_loop(det, timestamps, t_start, overflow, nch)
        for ts, ts_chunk in zip(timestamps_m, ts_chunks):
            ts.append(ts_chunk)

    # Compute acquisition duration
    meta['acquisition_duration'] = duration(timestamps_m, ts_unit)
    return None, timestamps_m, meta


def ni96ch_process(fname, out_path=None, chunksize=2**18, num_timestamps=-1,
                   debug=False, close=False, inner_loop=None, comp_filter=None,
                   progrbar_widget=True):
    """Sort timestamps per-ch and correct overflow in NI-96ch data.

    This function separates each single detector channel and corrects overflows.
    The 96 arrays are saved to a (hopefully temporary) HDF5 file. To create
    Photon-HDF5 files, channels pairs (i.e. Donor-Acceptor) need to be merged.
    This function auto-detects whether the file is saved by LabVIEW
    MultiCounterProject (so it has a 3-lines header and timestamps in
    big-endian order) or by LabVIEW FPGA_96ch project (no header, timestamps
    in little-endian order).

    Arguments:
        out_path (string or pathlib.Path or None): name of the ouput HDF5 file.
            If None, use same name as input file appending '_raw_temp.hdf5'.
        chunksize (int): input file is read in chunks (i.e. number of 32-bit
            words) of this size.
        num_timestamps (int): read at most `num_timestamps`. If negative read
            the whole file.
        close (bool): wether to close the output pytables file
        debug (bool): perform additional consistency checks.
        inner_loop (function or None): function to use in the inner loop
            for overflow correction of each chunk of timestamp.
        comp_filter (tables.Filters): compression filter for the pytables file.
        progrbar_widget (bool): If true display progress bar as a Jypyter
            notebook widget.

    Returns:
        A tuple of:

        - h5file (pytables file): the handle for the pytables file
        - timestamps_m (list): list of pytables timetamps arrays
        - meta (dict): metadata extracted from the file
    """
    fname = Path(fname)
    if inner_loop is None:
        inner_loop = _inner_loop2
    dt, endianess, meta = detectformat(fname)
    if num_timestamps < meta['num_timestamps']:
        num_timestamps = meta['num_timestamps']
    nbits = 24
    ts_max = 2**nbits
    nch = meta['nchannels']
    ts_unit = 1 / meta['clock_frequency']
    if out_path is None:
        out_path = Path(fname.parent, fname.stem + '_raw_temp.hdf5')
    out_path = Path(out_path)

    # Open file and position cursor after header
    f = open(fname, 'rb')
    f.seek(meta['offset'])

    # Output file
    if comp_filter is None:
        comp_filter = tables.Filters(complevel=6, complib='blosc')
    h5file = tables.open_file(str(out_path), mode="w", filters=comp_filter)
    for ch in range(nch):
        h5file.create_earray('/', 'timestamps%d' % ch, chunkshape=(chunksize,),
                             obj=np.array([], dtype=np.int64))

    # List of empty timestamps arrays in HDF5 file
    timestamps_m = [h5file.get_node('/timestamps%d' % ch) for ch in range(nch)]

    # Separate channels and correct overflow
    t_start = np.zeros(nch, dtype='int64')
    progressbar = tqdm_notebook if progrbar_widget else tqdm
    _iter = progressbar(iter_chunksize(num_timestamps, chunksize),
                        total=np.ceil(num_timestamps / chunksize))
    for chunksize in _iter:
        timestamps, det = _read_chunk(f, chunksize, dt, endianess, nbits)
        ts_chunks = inner_loop(det, timestamps, t_start, ts_max, nch)
        for ts, ts_chunk in zip(timestamps_m, ts_chunks):
            ts.append(ts_chunk)
            if debug:
                assert (np.diff(ts_chunk) > 0).all()

    # Compute acquisition duration
    meta['acquisition_duration'] = duration(timestamps_m, ts_unit)
    h5file.flush()
    if close:
        h5file.close()
    return h5file, meta


def ni96ch_process_spots(fname, out_path=None, chunksize=2**18, num_timestamps=-1, debug=False,
                         close=False, inner_loop=None, comp_filter=None,
                         progrbar_widget=True):
    """Sort timestamps per-spot and correct overflow in NI-96ch data.

    This function auto-detects whether the file is saved by LabVIEW
    MultiCounterProject (so it has a 3-lines header and timestamps in
    big-endian order) or by LabVIEW FPGA_96ch project (no header, timestamps
    in little-endian order).

    Arguments:
        fname (string or pathlib.Path): name of the input data file.
        out_path (string or pathlib.Path or None): name of the ouput HDF5 file.
            If None, use same name as input file changing the extension to hdf5.
        chunksize (int): input file is read in chunks (i.e. number of 32-bit
            words) of this size.
        num_timestamps (int): read at most `num_timestamps`. If negative read
            the whole file.
        close (bool): wether to close the output pytables file
        debug (bool): perform additional consistency checks.
        inner_loop (function or None): function to use in the inner loop
            for overflow correction of each chunk of timestamp.
        comp_filter (tables.Filters): compression filter for the pytables file.
        progrbar_widget (bool): If true display progress bar as a Jypyter
            notebook widget.

    Returns:
        A tuple of:

        - h5file (pytables file): the handle for the pytables file
        - meta (dict): metadata extracted from the file
    """
    fname = Path(fname)
    if inner_loop is None:
        inner_loop = _inner_loop2
    dt, endianess, meta = detectformat(fname)
    if num_timestamps < meta['num_timestamps']:
        num_timestamps = meta['num_timestamps']
    nbits = 24
    ts_max = 2**nbits
    spots = np.arange(48)
    #nch = 2 * spots.size
    ts_unit = 1 / meta['clock_frequency']
    if out_path is None:
        out_path = fname.with_suffix('.hdf5')
    out_path = Path(out_path)

    # Open file and position cursor after header
    f = open(fname, 'rb')
    f.seek(meta['offset'])

    # Output file
    if comp_filter is None:
        comp_filter = tables.Filters(complevel=6, complib='zlib')
    h5file = tables.open_file(str(out_path), mode="w", filters=comp_filter)
    for spot in spots:
        h5file.create_earray('/photon_data%d' % spot, 'timestamps',
                             createparents=True, chunkshape=(chunksize,),
                             obj=np.array([], dtype=np.int64))
        h5file.create_earray('/photon_data%d' % spot, 'detectors',
                             chunkshape=(chunksize,),
                             obj=np.array([], dtype=np.uint8))

    # List of empty timestamps arrays in HDF5 file
    timestamps_m, detectors_m = get_photon_data_arr(h5file, spots)

    # Separate channels and correct overflow
    t_start = np.zeros(2 * spots.size, dtype='int64')
    progressbar = tqdm_notebook if progrbar_widget else tqdm
    _iter = iter(progressbar(iter_chunksize(num_timestamps, chunksize),
                             total=np.ceil(num_timestamps / chunksize)))

    timestamps, det = _read_chunk(f, next(_iter), dt, endianess, nbits)
    prev_ts_chunks, prev_det_chunks = _inner_loop_spots(det, timestamps,
                                                        t_start, ts_max)
    ts_idx = chunksize
    for chunksize in _iter:
        timestamps, det = _read_chunk(f, chunksize, dt, endianess, nbits)
        ts_idx += chunksize
        ts_chunks, det_chunks = _inner_loop_spots(det, timestamps, t_start,
                                                  ts_max)
        last_ts_chunks, last_det_chunks = [], []
        for i, (ts, det) in enumerate(zip(timestamps_m, detectors_m)):
            last_two_ts_chunks = [prev_ts_chunks[i], ts_chunks[i]]
            last_two_det_chunks = [prev_det_chunks[i], det_chunks[i]]
            _fix_order(i, last_two_ts_chunks, last_two_det_chunks)
            ts.append(last_two_ts_chunks[0])
            det.append(last_two_det_chunks[0])
            prev_ts_chunks[i] = last_two_ts_chunks[1]
            prev_det_chunks[i] = last_two_det_chunks[1]
            last_ts_chunks.append(last_two_ts_chunks[1])
            last_det_chunks.append(last_two_det_chunks[1])
            if debug:
                assert (np.diff(ts_chunks[i]) > 0).all()

    # Save the last chunk for each spot
    for i, (ts, det) in enumerate(zip(timestamps_m, detectors_m)):
        ts.append(last_ts_chunks[i])
        det.append(last_det_chunks[i])

    # Compute acquisition duration
    meta['acquisition_duration'] = duration(timestamps_m, ts_unit)
    h5file.flush()
    if close:
        h5file.close()
    return h5file, meta


def _fix_order(ispot, two_ts_chunks, two_det_chunks):
    """Fix non-sorted timestamps in consecutive chunks of per-spot data.

    Each timestamp-chunk is monotonic but their concatenation may be
    non-monotonic. In the latter case, this function modifies in-place
    the input lists to fix the timestamps (and detectors) order.
    """
    assert len(two_ts_chunks) == 2
    for i in (0, 1):
        assert len(two_ts_chunks[i]) == len(two_det_chunks[i])
    # Check cross-chunk monotonicity
    if two_ts_chunks[1][0] < two_ts_chunks[0][-1]:
        ts_merged = np.hstack(two_ts_chunks)
        det_merged = np.hstack(two_det_chunks)
        sorted_index = ts_merged.argsort(kind='mergesort')
        ts_merged = ts_merged[sorted_index]
        det_merged = det_merged[sorted_index]
        size0 = len(two_ts_chunks[0])
        two_ts_chunks[:] = [ts_merged[:size0], ts_merged[size0:]]
        two_det_chunks[:] = [det_merged[:size0], det_merged[size0:]]


def merge_timestamps(t1, t2):
    tm = np.hstack([t1, t2])
    index_sort = tm.argsort(kind='mergesort')  # stable sorting
    tm = tm[index_sort]
    mask_t2 = np.hstack([np.zeros(t1.size, dtype=bool),
                         np.ones(t2.size, dtype=bool)])[index_sort]
    return tm, mask_t2


def save_timestamps_detectors_48ch(timestamps_m, h5file, chunksize=2**18,
                                   comp_filter=None, progrbar_widget=True):
    """Save per-spot timestamps/detectors from a list of per-ch timestamps.
    """
    if comp_filter is None:
        comp_filter = tables.Filters(complevel=6, complib='blosc')
    spots, D_ch, A_ch = get_spot_ch_map_48spots()
    progressbar = tqdm_notebook if progrbar_widget else tqdm
    for ch_a, ch_d in progressbar(zip(A_ch, D_ch), total=48):
        spot = ch_a
        td = timestamps_m[ch_d][:]
        ta = timestamps_m[ch_a][:]
        tm, a_em = merge_timestamps(td, ta)
        kws = dict(chunkshape=(chunksize,), filters=comp_filter)
        h5file.create_carray('/photon_data%d' % spot, 'timestamps', obj=tm,
                             createparents=True, **kws)
        h5file.create_carray('/photon_data%d' % spot, 'detectors', obj=a_em,
                             **kws)
    h5file.flush()
    ts_list , A_em = get_photon_data_arr(h5file, spots)
    detectors_ids = np.empty(96, dtype=np.uint8)
    detectors_ids[0::2] = D_ch
    detectors_ids[1::2] = A_ch
    spots = np.repeat(range(48), 2)
    return ts_list, A_em, detectors_ids, spots


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Utility functions
#

def get_spot_ch_map_48spots():
    # A-ch are [0..47] and assumed to be the spot number
    # D-ch are [48..95], the mapping from D-ch to spot number is below in D_ch
    A_ch = spots = np.arange(48)
    D_ch = np.arange(48).reshape(4, 12)[::-1].reshape(48) + 48
    return spots, D_ch, A_ch


def get_photon_data_arr(h5file, spots):
    """Return two lists with timestamps and detectors arrays from `h5file`.
    """
    ts_list = [h5file.get_node('/photon_data%d/timestamps' % ch) for ch in spots]
    A_em = [h5file.get_node('/photon_data%d/detectors' % ch) for ch in spots]
    return ts_list, A_em


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions to create a Photon-HDF5
#

def create_ph5data_smFRET_48spots(
        orig_filename, h5file, ts_unit, metadata,
        excitation_wavelengths=(532e-9,),
        detection_wavelengths = (580e-9,),
        software = 'LabVIEW MultiCounterProject/Multiple Counters UI v8_96 ch_Generic (NI-FPGA)',
        setup = None):
    """
    Create a `dict` suitable for saving a Photon-HDF5 file.
    """
    provenance = dict(filename=orig_filename, software=software)
    assert 'acquisition_duration' in metadata
    data = dict(
        _filename = orig_filename,
        provenance=provenance)
    data.update(metadata)
    data['acquisition_duration'] = np.round(data['acquisition_duration'], 1)

    if setup is None:
        setup = dict(
            num_pixels = 96,
            num_spots = 48,
            num_spectral_ch = 2,
            num_polarization_ch = 1,
            num_split_ch = 1,
            modulated_excitation = False,
            lifetime = False,
            excitation_wavelengths = excitation_wavelengths,
            excitation_cw = (False,),
            detection_wavelengths = detection_wavelengths,
            excitation_alternated = (False,)
            )

    if setup != 'skip':
        data['setup'] = setup

    return fill_photon_data_tables(data, h5file, ts_unit)


def fill_photon_data_tables(data, h5file, ts_unit, measurement_specs=None):
    """Fill the `data` dict with "photon_data" taken from h5file.
    """
    if measurement_specs is None:
        measurement_specs = dict(
            measurement_type = 'smFRET',
            detectors_specs = dict(spectral_ch1 = 0, spectral_ch2 = 1))

    ts_list, A_em = get_photon_data_arr(h5file, spots=range(48))

    for ich, (times, a_em) in enumerate(zip(ts_list, A_em)):
        data.update(
            {'photon_data%d' % ich:
             dict(
                timestamps = times,  # a pytables array!
                timestamps_specs = dict(timestamps_unit=ts_unit),
                detectors = a_em,    # a pytables array!

                measurement_specs = measurement_specs)})
    return data


def populate_metadata_smFRET_48spots(metadata, orig_filename, acq_duration):
    """Populate metadata for a smFRET-48spot setup (either single-laser or PAX).

    Automatically populate the "/setup" group for a smFRET-48spot setup.
    If length of `excitation_alternated` in the input metadata if 1, then
    single-laser (532nm) excitation is assumed. If the length is 2, then
    PAX is assumed.
    This function also fills: /sample/num_dyes, /provenance/filename,
    /provenance/software. /acquisition_duration is rounded to 1 decimal.
    It does not fill identity group with authors and/or affiliations.

    Arguments:
        metadata (dict): a nested dictionary representing some fields in a
            Photon-HDF5 file. It must contain a list/tuple of bools
            in `metadata['setup']['excitation_alternated']`.
        orig_filename (pathlib.Path or string): path of the original DAT file,
            to be stored in /provenance/filename.
        acq_duration (float): acquisition duration in seconds.

    Returns:
        A new dictionary (copy) with the complete metadata.
    """
    metadata = metadata.copy()
    setup = metadata['setup']
    if len(setup['excitation_alternated']) == 1:
        kind = 'single-laser smFRET'
    elif len(setup['excitation_alternated']) == 2:
        kind = 'PAX'
    else:
        raise ValueError('excitation_alternated should be of len 1 or 2.')

    default_setup = dict(
        num_spectral_ch = 2,
        num_polarization_ch = 1,
        num_split_ch = 1,
        lifetime = False,
        num_spots = 48,
        num_pixels = 96,
        detection_wavelengths = [580e-9, 660e-9],
        )
    if kind == 'PAX':
        # smFRET-PAX (532nm CW, 628nm alternated)
        default_setup.update(
            excitation_wavelengths = [532e-9, 628e-9],
            excitation_cw = [True, True],
            modulated_excitation = True,
            excitation_alternated = [False, True]
            )
    else:
        # Single-laser smFRET with 532nm excitation
        default_setup.update(
            excitation_wavelengths = [532e-9],
            excitation_cw = [True],
            modulated_excitation = False,
            excitation_alternated = [False],
            )

    # Fill-in only the setup fields not present in metadata
    for k in default_setup.keys():
        setup[k] = setup.get(k, default_setup[k])

    # Sample group
    if 'sample' in metadata:
        sample = metadata['sample']
        sample['num_dyes'] = len(sample['dye_names'].split(','))

    # Create or update provenance group
    sw = 'LabVIEW MultiCounterProject/Multiple Counters UI v8_96 ch_Generic (NI-FPGA)'
    provenance = metadata.get('provenance', dict())
    provenance.update(filename=str(orig_filename), software=sw)

    # Other metadata
    metadata['acquisition_duration'] = np.round(acq_duration, 1)

    return metadata
