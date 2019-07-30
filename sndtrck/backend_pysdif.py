import numpy as np
import pysdif
import array
import logging

logger = logging.getLogger("sndtrck")


try:
    import pysdif
    pysdif.sdif_init()
    AVAILABLE = True
    logger.debug("pysdif backend found")
except ImportError:
    logger.warning("pysdif backend not found")
    AVAILABLE = False
    

def is_available():
    # type: () -> bool
    return AVAILABLE


def aslist(l):
    if isinstance(l, list):
        return l
    return list(l)


def get_info():
    # type: () -> dict
    return {
        'name': 'pysdif',
        'analyze': False,
        'read_sdif': True,
        'write_sdif': True,
        'synthesize': False,
        'estimatef0': False,
    }


def write_sdif(outfile, matrices, labels=None, rbep=True, fadetime=0):
    if rbep:
        return write_rbep(outfile, matrices, labels)
    else:
        logger.error("1TRC is not supported yet")
        return False


def read_sdif(infile):
    s = pysdif.SdifFile(infile)
    sig = s.signature
    if sig == b"RBEP":
        return _read_rbep(s)
    elif sig == b"1TRK":
        return _read_1trck(s)
    else:
        raise IOError(f"Sdiffile with unknown signature: {sig}")


def _read_1trck(sdif):
    raise NotImplementedError("!!")


def _read_rbep(sdif):
    RBEP = pysdif.str2signature(b"RBEP")  # type: int
    prepartials = {}
    for frame in sdif:
        if frame.numerical_signature == RBEP:
            t = frame.time 
            sig, data = frame.get_matrix()
            for i in range(data.shape[0]):
                row = data[i]
                idx = int(row[0])
                row[5] += t   # offset + frametime
                p = prepartials.get(idx)
                if p:
                    p.append(row)
                else:
                    prepartials[idx]= [row]
        else:
            logger.debug(f"Skipping frame with signature: {frame.signature}")
    sdif.close()
    matrices = []
    for rows in prepartials.values():
        assert isinstance(rows, list)
        mtx = np.stack(rows)
        # N F A P B T -> T F A P B
        mtx[:,0] = mtx[:,5]
        mtx = mtx[:,:5]
        matrices.append(mtx)
    return matrices


def write_rbep(outfile, matrices, labels, rbep=True, **kws):
    # TODO: save labels
    matrices = aslist(matrices)
    assert isinstance(outfile, str)
    assert isinstance(matrices, list)
    assert isinstance(matrices[0], np.ndarray)
    assert isinstance(labels, list) or labels is None
    assert isinstance(rbep, bool)

    logger.debug("opening %s" % outfile)
    sdif = pysdif.SdifFile(outfile, "w")
    logger.debug("adding type")
    sdif.add_predefined_frametype(b"RBEP")
    numbps = [len(m) for m in matrices]
    sumbps = sum(numbps)
    # alltimes = np.concatenate([p.times for p in spectrum])
    alltimes = np.concatenate([m[:,0] for m in matrices])
    # allidx = np.concatenate([np.full((n,), i, dtype=int) for i, n in enumerate(numbps)])
    maxbps = max(numbps)
    indices = np.arange(maxbps, dtype=int)
    allidx = np.empty((sumbps,), dtype=int)
    pos = 0
    for i, n in enumerate(numbps):
        allidx[pos:pos+n] = i
        pos += n
    allbpidx = np.concatenate([indices[:n] for n in numbps])
    sortidx = np.argsort(alltimes)
    t0 = alltimes[sortidx[0]]
    sorted_idx = allidx[sortidx]
    sorted_bpidx = allbpidx[sortidx]
    N = len(matrices)
    frametime = t0
    present0 = array.array("b", [0]) * N
    present = present0[:]
    rowidx = -1
    bigmatrix = np.zeros((N, 6), dtype=float)
    for i in range(sortidx.shape[0]):
        idx = sorted_idx[i]
        bpidx = sorted_bpidx[i]
        # rbep: index freq amp phase bw offset
        # 1trc: index freq amp phase
        arr = matrices[idx]
        t = arr[bpidx, 0]
        if present[idx]:
            matrix = bigmatrix[:rowidx+1]
            sdif.new_frame_one_matrix("RBEP", frametime, "RBEP", matrix)
            frametime = t
            rowidx = 0
            present[:] = present0
        else:
            present[idx] = 1
            rowidx += 1
        assert rowidx < N
        bigmatrix[rowidx, 0] = idx
        bigmatrix[rowidx, 1] = arr[bpidx, 1]
        bigmatrix[rowidx, 2] = arr[bpidx, 2]
        bigmatrix[rowidx, 3] = arr[bpidx, 3]
        bigmatrix[rowidx, 4] = arr[bpidx, 4]
        bigmatrix[rowidx, 5] = t - frametime
    matrix = bigmatrix[:rowidx+1]    
    sdif.new_frame_one_matrix("RBEP", frametime, "RBEP", matrix)
    sdif.close()
