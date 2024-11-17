#! /usr/bin/env python
# coding:utf8


"""Library for querying the genome sequence stored as a FASTA.

**CAUTION** : verify which coordinate system you are using ; some functions expect 1-based closed-end
intervals ([start:end]) such as found in VCF format, GTF/GFF format, or FASTA indexing coordinates,
while you might be manipulating 0-based open-end coordinates ([start:end)) as found in BED format.
"""

from typing import Mapping, Union, Any
from typing_extensions import Self
import os

import numpy as np
import pandas as pd
import pyfaidx

import py2bit
import Bio.Seq
from Bio.SeqRecord import SeqRecord

from abc import ABC, abstractmethod

from enum import Enum


class Offset(Enum):
    LEFT = "left"
    RIGHT = "right"


def center_window_coordinates(
    midpoint_coord: int, windowsize: int, offset_side: Offset = Offset.LEFT
):
    """Return (start,end) coordinates of interval of size <windowsize> centered on <midpoint_coord>.

    <offset_side> option allows to deal with windows of even size :
    - 'left' indicates that the midpoint coordinate should be the
        starting point of the second half of the window
    - 'right' indicates that the midpoint coordinate should be the
        end point of the first half of the window

    e.g. with a windowsize of 4 and a midpoint coord of 2, 'left' will return the coordinates [0:3]
    while 'right' will return the coordinates [1:4].

    Args:
        midpoint_coord (int) : coordinate on which the window should be centered.
        windowsize (int) : size of window for which coordinates should be obtained.
        offset_side (str, ['left','right'], default='left') : which side to offset when <windowsize> is even

    Returns:
        tuple : (int, int) corresponding to the calculated [start:end] positions
                of the centered window. Note that the function does not apply any
                correction on these positions, so all values are possible, including
                negative values.
    """
    if not isinstance(windowsize, int) and windowsize > 1:
        raise ValueError("<windowsize> should be an integer>1")

    if windowsize % 2 == 0:
        # Situation where offset is needed.
        if offset_side == Offset.LEFT:
            left_size = int(windowsize / 2)
            right_size = left_size - 1

        else:
            right_size = int(windowsize / 2)
            left_size = right_size - 1

        start = midpoint_coord - left_size
        end = midpoint_coord + right_size

    else:
        # No correction needed : we set the same number of bases on each side
        # of the midpoint.
        start = midpoint_coord - int(np.floor(windowsize / 2))
        end = midpoint_coord + int(np.floor(windowsize / 2))

    return (start, end)


class IndexedSequences(ABC):
    @abstractmethod

    def fetch_sequence(self, seqname: str, start: int, end: int) -> str:
        ...

    @abstractmethod
    def chromsizes(self) -> Dict[str, int]:
        ...

    @abstractmethod
    def __enter__(self) -> Self:
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        ...

    @property
    def filepath(self) -> Union[str, os.PathLike]:
        return self._filepath

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def fileobj(self) -> Any:
        return self._fileobj

    def _check_file_status(self) -> None:
        if self.fileobj is None:
            raise OSError("File not opened. Use as context manager `with <instance> as <opened_instance>: ...` or call `<instance>.open()`.")

        if self.closed:
            raise OSError("File already closed.")


class TwoBitIndexedSequences(IndexedSequences):
    def __init__(
        self, twobit_path: Union[str, os.PathLike], force_open: bool = False
    ) -> None:
        self._filepath = twobit_path
        self._fileobj = None
        self._closed = False
        self._chromsizes = None

        if force_open:
            self.open()

    def open(self):
        self._fileobj = py2bit.open(str(self.filepath))

    def close(self):
        self.fileobj.close()
        self._closed = True

    def __enter__(self) -> Self:
        self._fileobj = py2bit.open(str(self.filepath))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.close()
        self._fileobj = None
        self._closed = False

    def fetch_sequence(self, seqname: str, start: int, end: int):
        """Call the py2bit method to fetch the sequence from the file. Expects 0-based open-end coordinates."""
        self._check_file_status()
        return self.fileobj.sequence(seqname, start, end)

    @property
    def chromsizes(self) -> Dict[str, int]:
        # Load the chromsizes structure.
        if self._chromsizes is None:
            self._check_file_status()
            self._chromsizes = self.fileobj.chroms()

        return self._chromsizes


class NewIndexedFasta(IndexedSequences):
    def __init__(
        self, fasta_path: Union[str, os.PathLike], chromsizes: Mapping[str, int], force_open: bool=False
    ) -> None:
        self.fasta_path = fasta_path
        self._chromsizes = chromsizes

    @property
    def chromsizes(self) -> Dict[str, int]:
        return self._chromsizes



class IndexedSequencesQuerier:
    def __init__(self, indexed_sequences: IndexedSequences) -> None:
        self.indexed_sequences = indexed_sequences

    @classmethod
    def from_indexed_fasta(
        cls, fasta_path: Union[str, os.PathLike], chromsizes: Mapping[str, int]
    ) -> Self:
        indexed_fasta = NewIndexedFasta(fasta_path, chromsizes, fillchar)
        return cls(indexed_fasta)

    @classmethod
    def from_twobit(cls, twobit_path: Union[str, os.PathLike]) -> Self:
        twobit_indexed = TwoBitIndexedSequences(twobit_path)
        return cls(twobit_indexed)

    def fetch_sequence(self, seqname: str, start: int, end: int) -> str:
        return self.indexed_sequences.fetch_sequence(seqname, start, end)

    @property
    def chromsizes(self) -> Dict[str, int]:
        return self.indexed_sequences.chromsizes


###############################################################################

class IndexedFasta:
    """ """

    _strand_converter = {
        "-": "-",
        -1: "-",
        "-1": "-",
        ".": "+",
        "+": "+",
        "1": "+",
        1: "+",
    }

    def __init__(
        self,
        fasta_path: Union[str, os.PathLike],
        chromsizes: Mapping[str, int],
        fillchar: str = "N",
    ) -> None:
        self.fasta_path = fasta_path
        self.chromsizes = chromsizes
        if not len(fillchar) == 1:
            raise ValueError(f"<fillchar> should be a character of length 1.")
        self.fillchar = fillchar

    def __repr__(self):
        return f"{type(self).__name__}('{self.fasta_path}')"

    def check_interval_in_chromsize(self, chrom: str, start: int, end: int):
        """Correct a 1-based closed-end genomic interval to fall within chrom. limits.

        Arguments:
            chrom (str)
            start (int) : 1-based start of the interval
            end (int) : 1-based included end of the interval
            chromsizes (dict) : map chromosome names to their total size

        Return:
            dict : corrected coordinates associated to the following keys :
                - new_start : corrected start (=start when no correction)
                - new_end : corrected end (=start when no correction)
                - left_start : N bases offset upstream the input start (0 when no correction)
                - right_end : N bases offset downstream the input end (0 when no correction)
        """
        if end < start:
            raise ValueError(f"end={end:,} < start={start:,}")

        chromsize = self.chromsizes[chrom]

        if start > chromsize:
            raise ValueError("Interval is beyond chrom size.")

        res = {
            "new_start": None,
            "new_end": None,
            "left_start": None,
            "right_end": None,
        }

        if start < 1:
            res["new_start"] = 1
            res["left_start"] = 1 - start
        else:
            res["new_start"] = start
            res["left_start"] = 0

        if end > chromsize:
            res["new_end"] = chromsize
            res["right_end"] = end - chromsize
        else:
            res["new_end"] = end
            res["right_end"] = 0

        if res["new_end"] < res["new_start"]:
            print(res)
            raise ValueError("New End < New start")

        return res

    def fetch_seq(self, chrom, start, end, strand):
        """Util to get a sequence (or its reverse complement)

        WARNINGS :
        - expects 1 based closed interval : [start-end]
        - start value <= 0 will raise a pyfaidx.FetchError
        - other incorrect intervals (end<start, interval beyond chromosome limits) return empty sequence

        Args:
            chrom (str)
            start (int)
            end (int)
            strand (int or str) : best if '-' or '+', but -1 is also handled as '-'

        Return:
            SeqRecord : sequence
        """

        # CAUTION : the indexing on the .fetch method is 1-based.
        # So to get the single base [100-101) in 0-based bed,
        # you have to run `fa.fetch(chrom, 101, 101)`

        with pyfaidx.Faidx(str(self.fasta_path)) as fa:
            seq_fetched = fa.fetch(chrom, start, end)

        strand = self._strand_converter[strand]
        if strand == "-":
            s = seq_fetched.reverse.complement
            RC_SUFFIX = "(RC)"
        else:
            s = seq_fetched
            RC_SUFFIX = ""

        # Rather than using pyfaidx ID, let's format the name a bit more.
        # seqid = str(s.long_name)
        seqid = f"{chrom}:{start}-{end}({strand}){RC_SUFFIX}"

        seq = SeqRecord(seq=Bio.Seq.Seq(s.seq), id=seqid, description="", name="")
        return seq

    def _update_seqid(self, seqid, start, end):
        """Util to update start end positions in seqid with new values"""
        newid = (
            f"{seqid.split(':')[0]}:{start}-{end}" f"({seqid.split('(', maxsplit=1)[1]}"
        )
        return newid

    def fetch_sequence(self, chrom, start, end, strand):
        """Util to get a sequence (or its reverse complement), spawning the entire interval.

        WARNINGS :
        - expects 1 based closed interval : [start-end].
        - incorrect sub-intervals (start<1 ; end > chromsome limits) will be filled with <self.fillchar>
          character.

        Args:
            chrom (str)
            start (int)
            end (int)
            strand (int or str): best if '-' or '+', but -1 is also handled as '-'

        Return:
            SeqRecord : sequence
        """
        # Need to check if within genomic interval.
        new_coords = self.check_interval_in_chromsize(chrom, start, end)
        seq = self.fetch_seq(
            chrom, new_coords["new_start"], new_coords["new_end"], strand
        )

        # Fill missing intervals
        extended_seq = self._flank_seq(
            seq, new_coords["left_start"], new_coords["right_end"]
        )

        return extended_seq

    def _flank_seq(self, seq, N_upstream, N_downstream):
        """Extend <seq> upstream and downstream with <self.fillchar>, and update <seq.id> accordingly."""
        newseq = self.fillchar * N_upstream + seq + self.fillchar * N_downstream
        seqid = seq.id
        start, end = map(int, seqid.split("(")[0].split(":")[1].split("-"))
        newseq.id = self._update_seqid(seqid, start - N_upstream, end + N_downstream)
        return newseq

    def fetch_flanked_sequence(
        self,
        chrom,
        start,
        end,
        strand,
        flank_upstream=0,
        flank_downstream=0,
        fill_flanks=False,
    ):
        """Query sequence for a genomic range, extended on flanks.

        Args:
            chrom (str)
            start (int) 1-based start interval
            end (int) : 1-based end interval, close-ended
            strand (str)
            windowsize (int) : the genomic range is extended by half this value on both sides.
            fill_flanks (bool) : if True, extend flanks with <self.fillchar> rather than querying the fasta.

        Return:
            str : sequence of size (end-start) + windowsize
        """
        if fill_flanks:
            # Get the sequence from the genome.
            seq = self.fetch_seq(chrom, start, end, strand)
            # Extend on both sides with Ns
            extended_seq = self._flank_seq(seq, flank_upstream, flank_downstream)

        else:
            # Here the flanking sequences are retrieved from the full sequence.
            # Caution : you may bypass chromosome limits here... would need to take care of this.
            start_extended = start - flank_upstream
            end_extended = end + flank_downstream
            extended_seq = self.fetch_sequence(
                chrom, start_extended, end_extended, strand
            )

        return extended_seq

    def fetch_centered_window(
        self, chrom, start, end, strand, windowsize, offset_side="left"
    ):
        """Fetch sequence of length <windowsize>, centered on the query interval.

        WARNINGS:
        - coordinates are expected to be 1-based, close-ended
        - if the extended window coordinates are beyond chromosome limits,
          missing sub-intervals will be filled with <self.fillchar> character
        - the center of the input (chrom,start,end) coordinates is calculated as
          `midpoint=int(start+np.floor((end-start)/2))`
        - to ensure the best location of the centered window, input the exact
          center position (start=end=middle point) rather than an interval


        Args:
            chrom (str)
            start (int)
            end (int)
            windowsize (int) : >1
            offset_side (str): for windows of even <windowsize>, choose which
                half of the window will **not** contain the mid-point.


        Returns:
            SeqRecord: sequence.
        """

        if not isinstance(windowsize, int) and windowsize > 1:
            raise ValueError("<windowsize> should be an integer>1")

        # Start/end should be 1 based, close-ended
        midpoint = start + int(np.floor((end - start) / 2))

        coord_window = center_window_coordinates(
            midpoint, windowsize, offset_side="left"
        )

        start_extended, end_extended = coord_window
        extended_seq = self.fetch_sequence(chrom, start_extended, end_extended, strand)

        if not len(extended_seq) == windowsize:
            raise ValueError(
                (
                    f"Retrieved sequence does not match expected "
                    f" <windowsize> : {len(extended_seq)}!={windowsize}"
                )
            )
        return extended_seq
