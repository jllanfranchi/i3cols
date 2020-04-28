# i3cols

Convert IceCube i3 files to columnar Numpy arrays &amp; operate on them.

## Installation

```
pip install i3cols
```

if using Python from CVMFS, the above will fail; you must use

```
pip install --user i3cols
```

and then the installed binary will be located at, e.g., `$HOME/.local/bin/i3cols`; add `$HOME/.local/bin` to your `PATH` to avoid having to type that full path every time (but reset your `PATH` if you switch out of CVMFS)

i3cols has two major pieces: i3 file data extraction and subsequent operations on the extracted data. If all you need to do is the former, and e.g. CVMFS-based Python is having problems installing Numba, you can get away with installing enum34 manually and then using `--no-deps` flag to `pip` such that Numba is not installed:

```
pip install --user enum34
pip install --user --no-deps i3cols
```

### For developers, or to get the very latest

Clone the repository and then perform an editable (`pip install -e`) installation:

```
git clone git@github.com:jllanfranchi/i3cols.git
pip install -e ./i3cols
```

## Examples

### Extracting data from I3 files

All command-line examples assume you are using BASH; adapt as necessary for your favorite shell.

Extract a few items from all Monte Carlo run 160000 files (nutau GENIE simulation performed for oscNext), concatenating into single column per item:

```bash
find /data/ana/LE/oscNext/pass2/genie/level7_v01.04/160000/ -name "oscNext*.i3*" | \
    i3cols extr_sep \
        --keys I3EventHeader I3MCTree I3MCWeightDict '*pulses*' \
        --index-and-concatenate-by subrun \
        --procs 20 \
        --outdir /data/user/jlanfranchi/columnar/genie/level7_v01.04/160000
        --tempdir /scratch/jlanfranchi
```

The above extracts each file within the `160000` directory separately into a sub-directory created in the `--tempdir`. Like columns are concatenated and the final result is placed in `--outdir`. Note that each file is interpreted as a subrun, as is the convention used for oscNext Monte Carlo simulation files. There will be an array called `subrun__categ_index.npy` created within the `--outdir` that identifies where each subrun's events are in the concatenated arrays. Note that the files input to the command are sorted by their subrun prior to being concatenated.

The `'*pulses*'` is glob-expanded (like `ls *` at the command line) and glob patterns match insensitive to case (while named keys are case-sensitive). You can also specify a file path to `--keys` that contains `\n`-separated key names and/or glob expressions (and/or other key files paths if you really want to get carried away)

If you completed the above and realize you also want the `I3GENIEResultDict`, then you can re-run the above but just specify that key (although specifying the above keys in addition to `I3GENIEResultDict` and not specifying the `--overwrite` flag should be no different, as already-extracted items will not be re-extracted). This process should then be much faster:

```bash
find /data/ana/LE/oscNext/pass2/genie/level7_v01.04/160000/ -name "oscNext*.i3*" | \
    i3cols extr_sep \
        --keys I3GENIEResultDict \
        --index-and-concatenate-by subrun \
        --procs 20 \
        --outdir /data/user/jlanfranchi/columnar/genie/level7_v01.04/160000
        --tempdir /scratch/jlanfranchi
```

Extract all keys from IC86.11 season except any key with the word "pulses" in it (glob patterns are matched ignoring case). All subrun files for a given run are combined transparently into one and then all runs are combined in the end into monolithic columns, with a `run__categ_index.npy` created in `outdir` that indexes the columns by run:

```bash
i3cols extr_season \
    --exclude-keys '*pulses*' \
    --index-and-concatenate \
    --gcd /tmp/i3/data/level7_v01.04/IC86.11/ \
    --outdir /tmp/columnar/data/level7_v01.04/IC86.11 \
    --compress \
    --procs 20 \
    /data/ana/LE/oscNext/pass2/data/level7_v01.04/IC86.11
```

More notes on the above examples:

* You can specify paths for extraction directly in the command (the last example), or you can pipe the paths to the function (the first two examples). The former is simple for specifying one or a few paths, but UNIX command lines are limited in total length, so the latter can be the only way to successfully pass all paths to i3cols.
* Optional compression of the resulting _column directories_ (a _column directory_ is a directory named after an item in the i3 frames, containing 1 or more `.npy` array files) can be performed after the extraction. The column directory is replaced with a `.npz` file. Memory mapping is not possible with `.npz` files, but significant compression ratios are often achievable.
* Extraction is performed in parallel where possible if `--procs` is provided with an argument > 1.


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


# Load just the I3MCTree (regardless of presence other columns), memory-mapped 
arrays, category_indexes = cols.load("/tmp/columnar/genie/level7_v01.04/160000", keys="I3MCTree", mmap=True)

# Look at the first event's "flattened" I3MCTree
i3mct_data = arrays["I3MCTree"]["data"]
i3mct_idx = arrays["I3MCTree"]["index"]
evt0_i3mct = i3mct_data[slice(*i3mct_idx[0])]
print(evt0_i3mct)

# Get the info!
tau_regen_evt_indices, tau_decay_products = get_tau_info_nb(**arrays["I3MCTree"])
```


## Motivation

IceCube .i3 files are formulated for arbitrary event-by-event processing of "events." The information that comprises an "event" can span multiple frames in the file, and the file must be read and written sequentially like linear tape storage (i.e., processing requires a finite state machine). This is well-suited to "online" and "real-time" processing of events.

Additionally, the IceTray, which is meant to read, process, and produce .i3 files, can process multiple files but is unwaware of file boundaries, despite the fundamental role that "number of files" plays in normalizing IceCube Monte Carlo simulation data.

Beyond collecting data, performing event splitting, and real-time algorithms, analysis often is most efficient and straightforward to work with specific features of events atomically: i.e., in a columnar fashion.

Two existing options are writing data to ROOT files and HDF5 files. The former requires a learning curve of its own outside of Python/Numpy, though possibly through projects like [Uproot](https://github.com/scikit-hep/uproot), such files could be more practical to use for Python-based analysis. HDF5 files suffer from the inability to be manipulated in parallel from Python and present an unessential layer of complexity around what we actually want and use: Numpy arrays of data.

**i3cols** allows working with IceCube data in columnar fashion without added complexity of libraries beyond Numpy, and both the columnar nature and the simplicity of the data representation allows more straightforward as well as new and novel ways of interacting with data which should be more natural and efficient for many common use-cases in Python/Numpy:


### Basics

1. Apply numpy operations directly to data arrays
2. Extract data columns to pass to machine learning tools directly or at most with simple indexing; e.g., to extract the azimuthal direction of all particles: `particles["dir"]["azimuth"]`.
3. Memory mapping allows chunked operations and arbitrary reading and/or writing to the same arrays from multiple processes (just make sure the processes don't try to write to the same elements).
4. New levels of processing entail adding new columns, without the need to duplicate all data that came before. This has the disadvantage that cuts that remove the vast majority of events result in columns that have the same number of rows as the pre-cut. However, the advantage to working this way is that it is trivial to go back to the lowest-level of processing and also to inspect how the cuts affect all variables contained at any level of processing. (A future goal is to also accommodate efficiently storing a subset of rows by creating a "sub-scalar" column that only contains rows where its or another column's "valid" array is True.)
5. Numpy arrays with structured dtypes can be passed directly to Numba for efficient looping with similar performance to compiled C++ code, as Numba is just-in-time (JIT)-compiled to machine code.
6. There is no dependency upon IceCube software once data has been extracted. This can be seen as a positive (portability) and a negative (IceCube software has been developed, in part, to perform analysis tasks).
7. If you think of a new item you want to extract from the source i3 files after already performing an extraction, it is _much_ faster and the process only yields the new column (rather than an entirely new file, as is the case with HDF5 extraction).
8. Basic operations like transferring specific frame items can be achieved with simple UNIX utilities (`cp`, `rsync`, etc.)
9. A future goal is to implement versioning with each column that is either explicitly accessible (if desired) or transparent (if desired) to users, such that different processing versions can live side-by-side without affecting one another.


### Flattening hierarchies for fast analysis without losing hierarchies

1. Source i3 files are invisible to analysis, if you want (the fact that data came from hundreds of thousands of small i3 files does not slow down analysis or grind cluster storage to a halt). If the concatenated arrays are too large to fit into memory, you can operate on arrays in arbitrary chunks of events and/or work directly on the data on disk via Numpy's built-in memory mapping (which is transparent to Numpy operations).
2. Source i3 files can be explicitly known to analysis if you want (via the Numpy arrays called in i3cols "category indexes").
3. Flattened data structures allow efficient operations on them. E.g., looking at all pulses is trivial without needing to traverse a hierarchy. But information about the hierarchy is preserved, so operating in that manner is still possible (and still very fast with Numpy and/or Numba).

### Data storage

There are two kinds of data currently supported:

1. **Scalar data**: One item per event. This includes arbitrarily complex dtypes, including a vector of N of a thing. The only requirement is that every event must also have exactly N of those things. This requires a _data_ array which has one entry per event. 
2. **Vector data**: Zero, one, or more of an item per event. E.g., pulses. This requires both _data_ and _index_ arrays. The index array has one entry per event which indicates the _start_ and _stop_ indices of that event's data. Meanwhile, _data_ can be arbitrarily long.

To accomodate arbitrary missing or invalid data, there can be an optional _valid_ array alongside the other arrays.

One **column** consists of the container (either a directory or `.npz` file) and the set of the above arrays it contains. The directory version of this looks on disk like, for example for _InIcePulses_ (which is vector data) with a valid array:
```
InIcePulses/
  |
  +-- data.npy
  |
  +-- valid.npy
  |
  +-- index.npy
```
If the above is compressed, it becomes a single `.npz `file: `InIcePulses.npz`. Within the file, the arrays can be accessed via their names "data", "valid", and/or "index". Here, we load all arrays that are present into a dictionary:
```python
with np.load("InIcePulses.npz") as npz:
    array_d = {name: npz.get(name, None) for name in ["data", "valid", "index"]}
```
Note there is a convenience function to do this (and load any category indices) for a single item or for an entire directory, transparent to compression: `arrays, category_indexes = i3cols.cols.load(dirpath)`.

#### Unsupported (or awkwardly supported) data types

1. **Mixture of types**: Scalar data where the item in one event has one type while another event (for that same item) has another type; similarly for vector data where dtype changes either within or across events. E.g., if the datatype contains a different-length-per-event vector as one field within the type (while all other parts of the type are constant-length). This can be accommodated (to varying degrees of inefficiency) by creating a type that contains all fields and just _one_ of the vector type, and then duplicating the fields that remain constant for every value in the vector.
2. **Many-fewer-than-1 valid item per event**: Suppose you start with simulation of N events and subsequently cut 90% of the simulated events. Operating on and storing all arrays dependent on that cut still uses arrays of length N, when only 10% of the data is valid. This is slower computationally and inefficient in terms of storage in memory and on disk.
3. **Metadata**: Currently, there is no way to handle this within the scope of i3cols. This is a TOOD item in the future. At least hints--if not complete information--should be stored so that generated columns can be re-generated in the future. Also, things like units of the contained data are valuable to keep with each column. Also there might be useful metadata to keep in the directory-of-columns. And all of this metadata should be kept around when columns are concatenated.

## See Also

The **i3cols** project was developed independently of but with the [Awkward Array project](https://github.com/scikit-hep/awkward-1.0) in mind. It is an eventual goal that the extraction of arrays and other IceCube-specific things from this project can remain, while the backend storage and manipulation of arrays can be done using that project.

A lower-level library to store data in memory (all data types null-able), [Apache Arrow](https://arrow.apache.org/) and its [Feather file format](https://arrow.apache.org/docs/python/feather.html) for storing on disk could be useful, where the Awkward Array project has been developed with this library in mind as well.

Finally, within IceCube software, the **tableio** module might be adaptable to produce similar output, and if i3cols can include metadata alongside the written arrays, that project has the ability to emit units and other useful metadata. For now, to prove the concept, the extraction part of i3cols is very straightforward (just not "automatic") as written, but utilizing tableio should be strongly considered in the future.
