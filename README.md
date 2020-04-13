# i3cols

Convert IceCube i3 files to columnar Numpy arrays &amp; operate on them.

## Motivation

IceCube .i3 files are formulated for arbitrary event-by-event processing of
"events."
The information that comprises an "event" can span multiple frames in the file,
and the file must be read and written sequentially like linear tape storage
(i.e., processing requires a finite state machine).
This is well-suited to "online" and "real-time" processing of events.

Additionally, the IceTray, which is meant to read, process, and produce .i3
files, can process multiple files but is unwaware of file boundaries, despite
the fundamental role that "number of files" plays in normalizing IceCube Monte
Carlo simulation data.

Beyond collecting data, performing event splitting, and real-time algorithms,
analysis often is most efficient and straightforward to work with specific
features of events atomically: i.e., in a columnar fashion.

**i3cols** allows working with IceCube data in this way, allowing new and novel
ways of interacting with data which should be more natural and efficient for
many common use-cases in Python/Numpy:


### Basics

1. Apply numpy operations directly to data arrays
2. Extract data columns to pass to machine learning tools directly
3. Memory mapping allows chunked operations and arbitrary reading and/or
   writing to the same arrays from multiple processes (just make sure they
   don't try to write to the same elements!)
4. New levels of processing entail adding new columns, without the need to
   duplicate all data that came before. This has the disadvantage that cuts
   that remove the vast majority of events result in columns that have the same
   number of rows as the pre-cut. However, the advantage to working this way is
   that it is trivial to go back to the lowest-level of processing and also to
   inspect how the cuts affect all variables contained at any level of
   processing.
5. Numpy arrays with structured dtypes can be passed directly to Numba for
   efficient looping with similar performance to compiled C++ code as Numba is
   just-in-time (JIT)-compiled.
6. There is no dependency upon IceCube software once they have been extracted.
7. If you think of a new item you want to extract from the source I3 files
   after already performing an extraction, it is _much_ faster and the process
   only yields the new column (rather than an entirely new file, as is the case
   with HDF5 extraction).
8. Basic operations like transferring specific frame items can be achieved with
   simple UNIX utilities (`cp`, `rsync`, etc.)
9. A future goal is to implement versioning with each column that is either
   explicitly accessible (if desired) or transparent (if desired) to users,
   such that different processing versions can live side-by-side without
   affecting one another.


### Flattening hierarchies

1. Source i3 files are invisible to analysis, if you want (the fact that data
    came from hundreds of thousands of small i3 files does not slow down
    analysis or grind cluster storage to a halt). If the concatenated arrays
    are too large to fit into memory, you can operate on arrays in arbitrary
    chunks of events and/or work directly on the data on disk via via Numpy's
    built-in memory mapping (which is transparent to Numpy operations)
2. Source i3 files can be explicitly known to analysis, if you want (via the
    Numpy arrays called in i3cals "category indexes")
3. Flattened datastructures allow efficient operations on them. E.g., looking
    at all pulses is trivial without needing to traverse a hierarchy. But
    information about the hierarchy is preserved, so operating in that manner
    is still possible (and still very fast with Numpy and/or Numba).


## Installation

```
pip install i3cols
```

### For developers

```
git clone git@github.com:jllanfranchi/i3cols.git
pip install -e i3cols
```

## Examples

### Extracting data from I3 files

All command-line examples assume you are using BASH; adapt as necessary for
your favorite shell.

Extract a few items from all Monte Carlo run 160000 files, concatenating into
single column per item :

```bash
find /tmp/i3/genie/level7_v01.04/160000/ -name "oscNext*.i3*" | \
    sort -V | \
    ~/src/i3cols/i3cols/cli.py extract_files_separately \
        --keys I3EventHeader I3MCTree I3MCWeightDict I3GENIEResultDict \
        --index-and-concatenate \
        --category-xform subrun \
        --procs 20 \
        --overwrite \
        --outdir /tmp/columnar/genie/level7_v01.04/160000 \
        --compress
```

Extract all keys from IC86.11 season. All subrun files for a given run are
combined transparently into one and then all runs are combined in the end into
monolithic columns, with a `run__category_index.npy` created in `outdir` that
indexes the columns by run:

```bash
~/src/i3cols/i3cols/cli.py extract_season \
    /tmp/i3/data/level7_v01.04/IC86.11/ \
    --index-and-concatenate \
    --gcd /data/icecube/gcd/ \
    --overwrite \
    --outdir /tmp/columnar/data/level7_v01.04/IC86.11 \
    --compress
```

Things to note in the above:

* You can specify paths on the command line, or you can pipe the paths to the
   function. The former is simple for specifying one or a few paths, but UNIX
   command lines are limited in total length, so the latter can be the only way to
   successfully pass all paths to i3cols (and UNIX pipes are nice ayway; note the
   numerical-version-sorting performed inline via
   `find <...> | sort -V | i3cols ...` in the first example).
* Optional compression of the resulting column directories (a directory + 1 or
    more npy arrays within) can be performed after the extraction. Memory mapping
    is not possible with the compressed files, but significant compression ratios
    are achievable.
* Extraction is performed in parallel where possible. Specify --procs to limit
    the number of subroccesses; otherwise, extraction (and
    compression/decompression) will attempt to use all cores (or hyperthreads,
    where available) on a machine.


### Working with the extracted data

Extracted data is output to Numpy arrays, possibly with structured Numpy dtypes.

```python
import numba

from i3cols import cols, phys

@numba.njit(fastmath=True, error_model="numpy")
def get_tau_info(data, index):
    """Return indices of events which exhibit nutau regeneration and return a
    dict of decay products of primary nutau.
    """

    tau_regen_evt_indices = numba.typed.List()
    tau_decay_products = numba.typed.Dict()
    for evt_idx, index_ in enumerate(index):
        flat_particles = data[index_["start"] : index_["stop"]]
        for flat_particle in flat_particles:
            if flat_particle["level"] == 0:
                if flat_particle["particle"]["pdg_encoding"] not in phys.NUTAUS:
                    break
            else:
                pdg = flat_particle["particle"]["pdg_encoding"]
                if flat_particle["level"] == 1:
                    if pdg not in tau_decay_products:
                        tau_decay_products[pdg] = 0
                    tau_decay_products[pdg] += 1
                if pdg in phys.NUTAUS:
                    tau_regen_evt_indices.append(evt_idx)
    return tau_regen_evt_indices, tau_decay_products


# Load arrays, memory-mapped
arrays, scalar_cat_indexes = cols.load("/tmp/columnar/genie/level7_v01.04/160000", mmap=True)

# Get the info!
tau_regen_evt_indices, tau_decay_products = get_tau_info_nb(**arrays["I3MCTree"])
```
