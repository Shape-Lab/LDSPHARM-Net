# Copyright 2021 Ilwoo Lyu, Seungbo Ha
# Contact: ilwoolyu@postech.ac.kr

"""File I/O operations for reading mesh and geometry data.

Supports multiple formats including FreeSurfer, VTK, SPHARM-Net,
and TIF templates for surface-based analysis.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _check_file(fname: str) -> None:
    """Raise FileNotFoundError if ``fname`` does not exist."""

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File not found: '{fname}'")


def read_curv(fname: str, nvert: int = -1) -> Tuple[np.ndarray, int]:
    """Read FreeSurfer's geometry file containing vertex-wise scalar data.

    Parameters
    ----------
    fname : str
        Path to the .curv file.
    nvert : int, optional
        Number of vertices to read. If -1, reads all available data.
        Default is ``-1``.

    Returns
    -------
    feat : numpy.ndarray, shape (n_vertex,)
        Vertex-wise scalar geometry values (e.g., curvature).
    fnum : int
        Number of faces in the associated mesh.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.

    Notes
    -----
    https://github.com/fieldtrip/fieldtrip/tree/master/external/freesurfer
    https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_curv.m

    They read binary files in big-endian.

    Examples
    --------
    >>> from spharmdnns.io import read_curv  # doctest: +SKIP
    >>>
    >>> curv, num_faces = read_curv("lh.curv")  # doctest: +SKIP
    >>> print(f"Loaded {len(curv)} vertex values, {num_faces} faces")  # doctest: +SKIP
    Loaded 163842 vertex values, 327680 faces (varies)
    """

    _check_file(fname)

    with open(fname, "rb") as fd:
        h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
        vnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

        if vnum == 0xFFFFFF:
            vnum, fnum, _ = np.fromfile(fd, dtype=np.dtype(">i4"), count=3)
            feat = np.fromfile(fd, dtype=np.dtype(">f4"), count=nvert)
        else:
            f0, f1, f2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            fnum = (f0.astype(int) << 16) + (f1.astype(int) << 8) + f2
            feat = np.fromfile(fd, dtype=np.dtype(">i2"), count=nvert) / 100

    return feat, fnum


def read_dat(fname: str, nvert: int = -1) -> np.ndarray:
    """Read SPHARM-Net's geometry file containing vertex-wise scalar data.

    Parameters
    ----------
    fname : str
        Path to the .dat file.
    nvert : int, optional
        Number of vertices to read. If -1, reads all available data.
        Default is ``-1``.

    Returns
    -------
    feat : numpy.ndarray, shape (n_vertex,)
        Vertex-wise scalar geometry values.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.

    Notes
    -----
    The function automatically detects label files by checking if the second-to-last
    part of the filename (before extension) is ``"label"``, and uses ``dtype=int16`` for
    labels and float64 for continuous features.

    Examples
    --------
    >>> from spharmdnns.io import read_dat  # doctest: +SKIP
    >>>
    >>> # Read a continuous feature file
    >>> curv_data = read_dat("lh.inflated.H.dat")  # doctest: +SKIP
    >>> print(f"Loaded {len(curv_data)} vertex values")  # doctest: +SKIP
    Loaded 163842 vertex values (varies)
    >>>
    >>> # Read a label file (will use int16 dtype)
    >>> labels = read_dat("lh.label.dat", nvert=163842)  # doctest: +SKIP
    >>> print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")  # doctest: +SKIP
    Labels shape: (163842,), dtype: int16
    """

    _check_file(fname)

    return np.fromfile(fname, dtype=np.int16 if fname.lower().split(".")[-2] == "label" else np.float64, count=nvert)


def read_txt(fname: str, nvert: int = -1) -> np.ndarray:
    """Read a geometry file in ASCII format containing vertex-wise scalar data.

    Parameters
    ----------
    fname : str
        Path to the .txt file.
    nvert : int, optional
        Number of vertices to read. If -1, reads all available data.
        Default is ``-1``.

    Returns
    -------
    feat : numpy.ndarray, shape (n_vertex,)
        Vertex-wise scalar geometry values.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.

    Notes
    -----
    The file should contain space-separated floating-point values.
    Values can be on a single line or multiple lines.

    Examples
    --------
    >>> from spharmdnns.io import read_txt  # doctest: +SKIP
    >>>
    >>> # Read space-separated ASCII file
    >>> data = read_txt("vertex_features.txt")  # doctest: +SKIP
    >>> print(f"Loaded {len(data)} values from ASCII file")  # doctest: +SKIP
    Loaded 163842 values from ASCII file (varies)
    >>>
    >>> # Read only first 1000 vertices
    >>> data_subset = read_txt("vertex_features.txt", nvert=1000)  # doctest: +SKIP
    >>> print(f"Loaded subset: {data_subset.shape}")  # doctest: +SKIP
    Loaded subset: (1000,)
    """

    _check_file(fname)

    return np.fromfile(fname, dtype=np.float64, sep=" ", count=nvert)


def read_annot(fname: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Read FreeSurfer's annotation file containing parcellation labels.

    Parameters
    ----------
    fname : str
        Path to the .annot file.

    Returns
    -------
    vID : numpy.ndarray, shape (n_vertex,)
        Vertex indices.
    label : numpy.ndarray, shape (n_vertex,)
        Vertex-wise label values (RGBA color codes).
    struct : list of str
        Names of brain structures/regions.
    structID : list of int
        IDs corresponding to the structures.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.

    Notes
    -----
    The annotation file format is specific to FreeSurfer and contains
    both vertex-to-label mappings and label-to-structure name mappings.
    The label values are RGBA color codes packed into 32-bit integers.

    References
    ----------
    https://github.com/fieldtrip/fieldtrip/tree/master/external/freesurfer

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>>
    >>> from spharmdnns.io import read_annot  # doctest: +SKIP
    >>>
    >>> # Read Desikan-Killiany atlas annotation
    >>> vID, labels, regions, region_ids = read_annot("lh.aparc.annot")  # doctest: +SKIP
    >>> print(f"Vertices: {len(vID)}, Regions: {len(regions)}")  # doctest: +SKIP
    Vertices: 163842, Regions: 36 (varies)
    >>> print(f"Region names: {regions[:3]}")  # doctest: +SKIP
    Region names: ['unknown', 'bankssts', 'caudalanteriorcingulate'] (varies)
    >>> print(f"Unique labels: {len(np.unique(labels))}")  # doctest: +SKIP
    Unique labels: 36 (varies)
    """

    _check_file(fname)

    with open(fname, "rb") as fd:
        size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
        buf = np.fromfile(fd, dtype=np.dtype(">i4"), count=2 * size).reshape(-1, 2)
        vID, label = buf[:, 0], buf[:, 1]

        struct = []
        structID = []

        has_struct = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)
        has_struct = has_struct[0] if has_struct.size > 0 else False

        if has_struct:
            nEntries = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]

            version = 0
            if nEntries <= 0:
                version = -nEntries
                if version != 2:  # pragma: no cover
                    raise ValueError(f"Unsupported FreeSurfer annotation version: expected version 2, got {version}")
                nEntries = np.fromfile(fd, dtype=">i4", count=1)

            size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
            buf = np.fromfile(fd, dtype=np.dtype(">i1"), count=size)
            # orig_tab = buf[:-1].tobytes().decode("ascii")

            if version == 2:
                nEntries = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]

            for _ in range(nEntries):
                if version == 2 and np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0] < 0:  # pragma: no cover
                    raise ValueError("Invalid annotation entry: entry index is negative")

                size = np.fromfile(fd, dtype=np.dtype(">i4"), count=1)[0]
                buf = np.fromfile(fd, dtype=np.dtype(">i1"), count=size)
                struct.append(buf[:-1].tobytes().decode("ascii"))
                buf = np.fromfile(fd, dtype=np.dtype(">i4"), count=4)
                structID.append(buf[0] + (buf[1] << 8) + (buf[2] << 16))

    return vID.astype(int), label.astype(int), struct, structID


def read_feat(fname: str, nvert: int = -1) -> np.ndarray:
    """Read a geometry file containing vertex-wise scalar data, auto-detecting format.

    Parameters
    ----------
    fname : str
        Path to the geometry file (.txt, .dat, or .curv).
    nvert : int, optional
        Number of vertices to read. If -1, reads all available data.
        Default is ``-1``.

    Returns
    -------
    feat : numpy.ndarray, shape (n_vertex,)
        Vertex-wise scalar geometry values.

    Notes
    -----
    This is a convenience function that dispatches to the appropriate
    reader based on file extension.

    Supported formats:

    - ``.txt`` -- read_txt (ASCII)
    - ``.dat`` -- read_dat (binary)
    - ``.curv`` or other -- read_curv (FreeSurfer binary)

    Examples
    --------
    >>> from spharmdnns.io import read_feat  # doctest: +SKIP
    >>>
    >>> # Automatically detects format from extension
    >>> curv = read_feat("lh.curv")  # FreeSurfer format  # doctest: +SKIP
    >>> sulc = read_feat("lh.sulc.dat")  # SPHARM-Net binary format  # doctest: +SKIP
    >>> thick = read_feat("thickness.txt")  # ASCII format  # doctest: +SKIP
    >>> print(f"Loaded features: curv={curv.shape}, sulc={sulc.shape}")  # doctest: +SKIP
    Loaded features: curv=(163842,), sulc=(163842,) (varies)
    >>>
    >>> # Read partial data
    >>> feat_partial = read_feat("lh.curv", nvert=10000)  # doctest: +SKIP
    >>> print(f"Partial load: {feat_partial.shape}")  # doctest: +SKIP
    Partial load: (10000,)
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".txt":
        feat = read_txt(fname, nvert)
    elif ext == ".dat":
        feat = read_dat(fname, nvert)
    else:
        feat, _ = read_curv(fname, nvert)

    return feat


def read_vtk(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a VTK mesh file containing 3D geometry.

    Parameters
    ----------
    fname : str
        Path to the .vtk file.

    Returns
    -------
    v : numpy.ndarray, shape (n_vertex, 3)
        3D coordinates of mesh vertices.
    f : numpy.ndarray, shape (n_face, 3)
        Triangle face indices, each row contains 3 vertex indices.

    Notes
    -----
    Supports both VTK legacy format (version < 5) and VTK version 5+ format.
    Automatically detects ASCII vs binary encoding from the file header.
    Only triangle meshes (POLYGONS with 3 vertices per face) are supported.

    Examples
    --------
    >>> from spharmdnns.io import read_vtk  # doctest: +SKIP
    >>>
    >>> # Read VTK mesh (ASCII or binary)
    >>> vertices, faces = read_vtk("sphere.vtk")  # doctest: +SKIP
    >>> print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")  # doctest: +SKIP
    Mesh: 163842 vertices, 327680 faces (varies)
    >>> print(f"Bounding box: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")  # doctest: +SKIP
    Bounding box: min=[-1. -1. -1.], max=[1. 1. 1.] (varies)
    >>>
    >>> # Check mesh integrity
    >>> assert faces.max() < len(vertices), "Face indices exceed vertex count"  # doctest: +SKIP
    """

    _check_file(fname)

    with open(fname, "rb") as fd:
        lines = iter(l for l in fd)

        ver = next(d for d in lines if b"Version" in d)
        ver = float(ver.split()[-1])

        is_binary = next(d.strip() for d in lines if d.strip() in [b"ASCII", b"BINARY"]) == b"BINARY"
        sep = "" if is_binary else " "

        nVert = next(d for d in lines if b"POINTS" in d)
        nVert = int(nVert.split()[1])
        dtype = np.dtype(">f4") if is_binary else float
        v = np.fromfile(fd, dtype=dtype, count=nVert * 3, sep=sep).reshape(nVert, 3)

        nFace = next(d for d in lines if b"POLYGONS" in d)
        nFace = int(nFace.split()[1])
        if ver < 5:
            dtype = np.dtype(">i4") if is_binary else int
            f = np.fromfile(fd, dtype=dtype, count=nFace * 4, sep=sep).reshape(nFace, 4)
            f = f[:, 1:]
        else:
            dtype = np.dtype(">i8") if is_binary else int
            nFace -= 1
            next(d for d in lines if b"CONNECTIVITY" in d)
            f = np.fromfile(fd, dtype=dtype, count=nFace * 3, sep=sep).reshape(nFace, 3)

    return v, f


def read_surf(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read FreeSurfer's mesh file containing 3D geometry.

    Parameters
    ----------
    fname : str
        Path to the surface file.

    Returns
    -------
    vertex_coords : numpy.ndarray, shape (n_vertex, 3)
        3D coordinates of mesh vertices.
    faces : numpy.ndarray, shape (n_face, 3)
        Triangle face indices, each row contains 3 vertex indices.

    Raises
    ------
    FileNotFoundError
        If ``fname`` does not exist.
    Exception
        If the file format is not recognized (unsupported magic number).

    Notes
    -----
    Supports three FreeSurfer surface formats identified by magic numbers.

    Supported formats:

    - ``0xFFFFFF`` -- Triangle surface (old format, scaled by 1/100)
    - ``0xFFFFFD`` -- Quad surface (old format, experimental)
    - ``0xFFFFFE`` -- Triangle surface (new format with metadata)

    Examples
    --------
    >>> from spharmdnns.io import read_surf  # doctest: +SKIP
    >>>
    >>> # Read FreeSurfer surface files
    >>> vertices, faces = read_surf("lh.white")  # White matter surface  # doctest: +SKIP
    >>> print(f"White matter: {vertices.shape[0]} vertices, {faces.shape[0]} faces")  # doctest: +SKIP
    White matter: 163842 vertices, 327680 faces (varies)
    >>>
    >>> # Read pial surface
    >>> v_pial, f_pial = read_surf("lh.pial")  # doctest: +SKIP
    >>>
    >>> # Read inflated surface
    >>> v_inflated, f_inflated = read_surf("lh.inflated")  # doctest: +SKIP
    """

    _check_file(fname)

    with open(fname, "rb") as fd:
        h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
        magic = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

        if (magic == 0xFFFFFF) | (magic == 0xFFFFFD):
            h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            vnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

            h0, h1, h2 = np.fromfile(fd, dtype=np.dtype("B"), count=3)
            fnum = (h0.astype(int) << 16) + (h1.astype(int) << 8) + h2

            vertex_coords = np.fromfile(fd, dtype=np.dtype(">i2"), count=3 * vnum) / 100
            vertex_coords = vertex_coords.reshape(-1, 3)
            arr = np.fromfile(fd, dtype=np.dtype("B"), count=12 * fnum).astype(int)
            faces = (arr[0::3] << 16) + (arr[1::3] << 8) + arr[2::3]
            faces = faces.reshape(-1, 4)

            return vertex_coords.astype(float), faces.astype(int)

        elif magic == 0xFFFFFE:
            fd.readline()
            fd.readline().strip()

            vnum, fnum = np.fromfile(fd, dtype=np.dtype(">i4"), count=2)
            vertex_coords = np.fromfile(fd, dtype=np.dtype(">f4"), count=3 * vnum)
            faces = np.fromfile(fd, dtype=np.dtype(">i4"), count=3 * fnum)

            vertex_coords = vertex_coords.reshape(vnum, 3)
            faces = faces.reshape(fnum, 3)

            return vertex_coords.astype(float), faces.astype(int)

    raise ValueError(
        f"Unrecognized surface file format for '{fname}': "
        "file does not match any supported format (FreeSurfer, VTK)"
    )


def read_mesh(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a mesh file containing 3D geometry, auto-detecting format.

    Parameters
    ----------
    fname : str
        Path to the mesh file (.vtk or FreeSurfer surface).

    Returns
    -------
    v : numpy.ndarray, shape (n_vertex, 3)
        3D coordinates of mesh vertices.
    f : numpy.ndarray, shape (n_face, 3)
        Triangle face indices, each row contains 3 vertex indices.

    Examples
    --------
    >>> from spharmdnns.io import read_mesh  # doctest: +SKIP
    >>>
    >>> vertices, faces = read_mesh("lh.white")  # doctest: +SKIP
    >>> print(f"Mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")  # doctest: +SKIP
    Mesh: 163842 vertices, 327680 faces (varies)
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".vtk":
        v, f = read_vtk(fname)
    else:
        v, f = read_surf(fname)

    return v, f

def write_vtk(
    fname: str, v: np.ndarray, f: np.ndarray, prop: Optional[Dict[str, np.ndarray]] = None, binary: bool = False
) -> None:
    """Write 3D mesh with optional vertex data to VTK format (VTK v3.0).

    Parameters
    ----------
    fname : str
        Output file path.
    v : numpy.ndarray, shape (n_vertex, 3)
        3D coordinates of mesh vertices.
    f : numpy.ndarray, shape (n_face, 3)
        Triangle face indices, each row contains 3 vertex indices.
    prop : dict of numpy.ndarray, optional
        Vertex-wise scalar data, each array shape (n_vertex,).
        Default is ``None``.
    binary : bool, optional
        Whether to write in binary format (default: ASCII).
        Default is ``False``.

    Notes
    -----
    Supports VTK legacy format version 3.0. Binary format uses big-endian
    encoding for cross-platform compatibility. Optional vertex properties
    are stored as FIELD ScalarData.

    Examples
    --------
    >>> import numpy as np  # doctest: +SKIP
    >>>
    >>> from spharmdnns.io import write_vtk  # doctest: +SKIP
    >>>
    >>> # Create simple mesh
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # doctest: +SKIP
    >>> faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)  # doctest: +SKIP
    >>>
    >>> # Write ASCII VTK
    >>> write_vtk("mesh.vtk", vertices, faces)  # doctest: +SKIP
    >>>
    >>> # Write with vertex properties
    >>> props = {'curvature': np.random.randn(4), 'thickness': np.random.rand(4)}  # doctest: +SKIP
    >>> write_vtk("mesh_with_props.vtk", vertices, faces, prop=props)  # doctest: +SKIP
    >>>
    >>> # Write binary VTK (smaller file size)
    >>> write_vtk("mesh_binary.vtk", vertices, faces, binary=True)  # doctest: +SKIP
    """

    len_v = v.shape[0]
    len_f = f.shape[0]

    with open(fname, "wb") as fd:
        fd.write("# vtk DataFile Version 3.0\nvtk output\n".encode())
        fd.write(f"{'ASCII' if not binary else 'BINARY'}\n".encode())
        fd.write("DATASET POLYDATA\n".encode())
        fd.write(f"POINTS {len_v} float\n".encode())

        if not binary:
            for row in v:
                fd.write(f"{row[0]} {row[1]} {row[2]}\n".encode())
            fd.write(f"POLYGONS {len_f} {len_f * 4}\n".encode())
            for row in f:
                fd.write(f"3 {row[0]} {row[1]} {row[2]}\n".encode())
            if prop is not None:
                fd.write(f"POINT_DATA {len_v}\n".encode())
                fd.write(f"FIELD ScalarData {len(prop)}\n".encode())
                for key in prop.keys():
                    fd.write(f"{key} 1 {len_v} float\n".encode())
                    val = prop[key]
                    for num in val:
                        fd.write(f"{num}\n".encode())
        else:
            fd.write(v.astype(">f4").tobytes())
            fd.write(f"POLYGONS {len_f} {len_f * 4}\n".encode())
            fd.write(np.hstack((np.full((f.shape[0], 1), 3), f)).astype(">i4").tobytes())
            if prop is not None:
                fd.write(f"POINT_DATA {len_v}\n".encode())
                fd.write(f"FIELD ScalarData {len(prop)}\n".encode())
                for key in prop.keys():
                    fd.write(f"{key} 1 {len_v} float\n".encode())
                    val = prop[key]
                    fd.write(val.astype(">f4").tobytes())
