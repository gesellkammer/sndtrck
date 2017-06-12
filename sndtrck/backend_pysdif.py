import numpy as np
import pysdif
import array
from .log import get_logger
logger = get_logger()

try:
    import pysdif
    pysdif.sdif_init()
    AVAILABLE = True
    logger.info("pysdif backend found")
except ImportError:
    logger.info("pysdif backend not found")
    AVAILABLE = False
    

def is_available():
    # type: () -> bool
    return AVAILABLE


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


def write_sdif(matrices, labels, outfile, rbep=True, fadetime=0):
    if rbep:
        return write_rbep(matrices, labels, outfile)
    else:
        logger.error("1TRC is not supported yet")
        return False


def write_rbep(matrices, labels, outfile):
    # TODO: save labels
    sdif = pysdif.SdifFile(outfile, "w")
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
        bigmatrix[rowidx, 0] = idx
        bigmatrix[rowidx, 1] = arr[bpidx, 1]
        bigmatrix[rowidx, 2] = arr[bpidx, 2]
        bigmatrix[rowidx, 3] = arr[bpidx, 3]
        bigmatrix[rowidx, 4] = arr[bpidx, 4]
        bigmatrix[rowidx, 5] = t - frametime
    matrix = bigmatrix[:rowidx+1]    
    sdif.new_frame_one_matrix("RBEP", frametime, "RBEP", matrix)
    sdif.close()
