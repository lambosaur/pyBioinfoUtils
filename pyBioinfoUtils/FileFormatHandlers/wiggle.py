# See https://genome.ucsc.edu/goldenPath/help/wiggle.html
# See https://www.ensembl.org/info/website/upload/wig.html

# Summary of properties:
#
# 1. A wiggle file contains one or more blocks, each containing a declaration line followed by lines defining data elements.
# 2. Wiggle data elements must be equally sized = in a block, all positions correspond to intervals of the same size.
# 3. VariableStep block: for data points with irregular spacing between data points.
# 4. FixedStep block: for data points with regular spacing between data points.
#
# Additional notes:
#
# - A wiggle file can contain multiple blocks, each with its own declaration line.
# - A wiggle file can contain blocks of different types.
# - A wiggle file likely contains parameters for visualization in a genome browser, as well as comments (lines starting with '#').
# - A wiggle file does not handle strand information. Strand-specific signal tracks are handled through dedicated separate files.
#


import collections
import re
import warnings
from abc import ABC, abstractmethod
from typing import IO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


class WiggleBlockTypeError(TypeError):
    pass


class VariableStepPositionError(ValueError):
    pass


class VariableStepParseError(ValueError):
    pass


class FixedStepParseError(ValueError):
    pass


class WiggleBlock(ABC):
    def __init__(self, *, chrom: str, values: Sequence[Union[int, float]], span: Optional[int] = None):
        self._chrom = chrom
        self._span = span
        self._values = tuple(values)

    _block_type = "wiggleBlock"

    @property
    def block_type(cls) -> str:
        return cls._block_type

    @property
    def chrom(self) -> str:
        return self._chrom

    @property
    def span(self) -> Optional[int]:
        return self._span

    @property
    def values(self) -> Tuple[Union[int, float], ...]:
        return self._values

    @property
    @abstractmethod
    def start(self) -> int:
        pass

    @property
    @abstractmethod
    def positions(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def header(self) -> str:
        pass

    def _check_span(self):
        if self.span and self.span > 100:
            warnings.warn("Span is greater than 100, consider using BedGraph format as per UCSC recommandation.")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"position": self.positions, "value": self.values})

    @abstractmethod
    def _items(self) -> Tuple:
        pass

    def __iter__(self) -> Iterable[object]:
        for item in self._items():
            yield item

    def _format_item_value(self, value: Union[int, float], precision: int) -> str:
        try:
            if isinstance(value, int):
                return f"{value:.d}"

            elif isinstance(value, float):
                return "{{value:.{precision}f}}".format(precision=precision).format(value=value)
            else:
                return str(value)
        except TypeError as e:
            raise TypeError(f"Error formatting value {value} with precision {precision}") from e

    @abstractmethod
    def _item_to_string(self, item: object, precision: int) -> str:
        pass

    def to_string_iterable(self, nmax: Optional[int] = None, precision: int = 9) -> Iterable[str]:
        yield self.header
        for i, item in enumerate(self.__iter__()):
            yield self._item_to_string(item, precision)
            if i == nmax:
                break

    def to_string(self, nmax: Optional[int] = None, precision: int = 9) -> str:
        return "\n".join(self.to_string_iterable(nmax, precision))

    def __str__(self) -> str:
        return self.to_string(nmax=3, precision=3) + "\n" + f"Total of {len(self.values):,} items."

    def __repr__(self) -> str:
        return self.to_string(nmax=None, precision=9)


class VariableStepBlock(WiggleBlock):
    def __init__(
        self,
        *,
        chrom: str,
        positions: Sequence[int],
        values: Sequence[Union[int, float]],
        span: Optional[int] = None,
    ):
        if not len(values) == len(positions):
            raise ValueError("positions and values must have the same length")

        super().__init__(chrom=chrom, values=values, span=span)

        self._positions = tuple(positions)
        self._check_span()
        self._validate_positions()

    _interval_type = "variableStep"

    @property
    def header(self) -> str:
        header = f"{self.block_type} chrom={self.chrom}"
        if self.span:
            return header + f" span={self.span}"
        return header

    @property
    def positions(self) -> Tuple[int, ...]:
        return self._positions

    @property
    def start(self) -> int:
        return self.positions[0]

    def _validate_positions(self):
        # Verify that the positions are greater than 0.
        if not self.positions[0] > 0:
            raise VariableStepPositionError("Wiggle positions are 1-relative, positions must be greater than 0")

        # Verify that the positions are sorted, monotonically increasing and unique
        for i in range(1, len(self.positions)):
            if self.positions[i] <= self.positions[i - 1]:
                raise VariableStepPositionError("Positions must be sorted and monotonically increasing")

            # NOTE: contiguity is not required. i.e. we should not test for exact equality to span.
            if self.span and ((self.positions[i] - self.positions[i - 1]) < self.span):
                raise VariableStepPositionError(
                    (
                        f"Error for position at index {i}: distance to previous position "
                        "should be greater or equal to the <span> value"
                    )
                )

    def _items(self) -> Iterable[Tuple[int, Union[int, float]]]:
        return zip(self.positions, self.values)

    def _item_to_string(self, item: Tuple[int, Union[int, float]], precision: int) -> str:
        position = item[0]
        value = item[1]
        return f"{position}" + self._format_item_value(value=value, precision=precision)


class FixedStepBlock(WiggleBlock):
    def __init__(
        self,
        *,
        chrom: str,
        start: int,
        step: int,
        values: Sequence[Union[int, float]],
        span: Optional[int] = None,
    ):
        super().__init__(chrom=chrom, values=values, span=span)
        self._start = start
        self._step = step

    _block_type = "fixedStep"

    @property
    def start(self) -> int:
        return self._start

    @property
    def step(self) -> int:
        return self._step

    @property
    def positions(self) -> Tuple[int, ...]:
        return tuple([self._start + self._step * i for i in range(len(self.values))])

    @property
    def header(self) -> str:
        header = f"{self.block_type} chrom={self.chrom} start={self.step} step={self.step}"
        if self.span:
            return header + f" span={self.span}"
        return header

    def _items(self) -> Iterable[Union[int, float]]:
        return iter(self.values)

    def _item_to_string(self, item: Union[int, float], precision: int = 9) -> str:
        return self._format_item_value(value=item, precision=precision)


class _ParsedHeader:
    def __init__(self, chrom: str, span: Optional[int], start: int, step: int):
        self._chrom = chrom
        self._span = span
        self._start = start
        self._step = step

    @property
    def chrom(self) -> str:
        return self._chrom

    @property
    def span(self) -> Optional[int]:
        return self._span

    @property
    def start(self) -> int:
        return self._start

    @property
    def step(self) -> int:
        return self._step


# work on a StringIO object
class WiggleParser:
    def __init__(self, stream: IO, encoding: str = "utf8"):
        self._stream = stream
        self._encoding = encoding

    @classmethod
    def from_stream(cls, stream: IO, encoding: str = "utf8") -> Iterable[WiggleBlock]:
        instance = cls(stream=stream, encoding=encoding)
        return instance.parse()

    def parse(self) -> Iterable[WiggleBlock]:
        # Iterate over lines of the stream.
        # when first part of the line match either "variableStep" or "fixedStep"
        # start accumulating the lines in a list until the next declaration line.
        # Then, yield an initialized WiggleBlock object.

        buffering = False
        line_buffer = collections.defaultdict(list)

        warned_about_comments = False
        warned_about_track_info_lines = False

        for line in self._stream:
            try:
                line = line.decode(self._encoding)
            except AttributeError:
                pass
            except UnicodeDecodeError as e:
                raise e

            # Here : check if we have a header for a block. This uses the
            # dictionary class property matching WiggleBlock._block_type string to
            # block parsers.
            for block_type in self._wiggle_block_type_to_parser:
                if line.startswith(block_type):
                    # Start of a new block
                    if buffering:
                        # This means there was a previous block.
                        yield self._parse_block(lines=line_buffer, block_type=block_type)
                        line_buffer = collections.defaultdict(list)
                    line_buffer["header"] = line
                    buffering = True
                    break
            else:
                if line.startswith("#"):
                    if not warned_about_comments:
                        warnings.warn("Comment line found in wiggle file. Not handled.", stacklevel=2)
                        warned_about_comments = True

                elif line.startswith("track"):
                    if not warned_about_track_info_lines:
                        warnings.warn("Track line found in wiggle file. Not handled.", stacklevel=2)
                        warned_about_track_info_lines = True
                else:
                    # Likely a data line.
                    # Need to test for numerical value conversion to make sure the
                    # line is a data line.
                    # Here: test on float values is done by replacing the first "." dot encountered
                    # by an empty string, to enable using the "isdigit" method.
                    # TODO: improve the test?
                    # TODO: is there any situation where we expect these values to be non-numeric?
                    # e.g. the format is not strictly enforced?
                    if line.strip().split(" ")[0].isdigit() or line.strip().split(" ")[0].replace(".", "", 1).isdigit():
                        line_buffer["data"].append(line)
                    else:
                        raise ValueError(f"Unrecognized line: {line}")

        # Yield the last block.
        if buffering:
            yield self._parse_block(lines=line_buffer, block_type=block_type)

    @property
    def _wiggle_block_type_to_parser(cls) -> Dict[str, Callable]:
        wiggle_block_types = {
            str(VariableStepBlock.block_type): cls._parse_variablestep_block,
            str(FixedStepBlock.block_type): cls._parse_fixedstep_block,
        }
        return wiggle_block_types

    def _parse_block(self, lines: Dict[str, List[str]], block_type: str) -> WiggleBlock:
        if not len(lines["header"]) == 1:
            raise ValueError("Block header must contain a single line")

        if not len(lines["data"]) > 0:
            raise ValueError("Block data must contain at least one line")

        parser = self._wiggle_block_type_to_parser[block_type]
        return parser(lines=lines)

    def _parse_header(self, header_line: str) -> _ParsedHeader:
        chrom = re.search(r"chrom=(\w+)", header_line)
        if chrom:
            chrom = chrom.group(1)
        else:
            raise ValueError("Chrom not found in header")

        span = re.search(r"span=(\d+)", header_line)
        if span is not None:
            try:
                span = int(span.group(1))
            except ValueError as e:
                raise ValueError(f"Span value must be an integer (parsed: {span})") from e

        start = re.search(r"start=(\d+)", header_line)
        if start is None:
            raise ValueError("Start not found in header")
        else:
            try:
                start = int(start.group(1))
            except ValueError as e:
                raise ValueError(f"Start value must be an integer (parsed: {start})") from e

        step = re.search(r"step=(\d+)", header_line)
        if step is not None:
            try:
                step = int(step.group(1))
            except ValueError as e:
                raise ValueError(f"Step value must be an integer (parsed: {step})") from e
        else:
            step = -1

        parsed_header = _ParsedHeader(chrom=chrom, span=span, start=start, step=step)

        return parsed_header

    def _parse_variablestep_block(self, lines: Dict[str, List[str]]) -> VariableStepBlock:
        parsed_header = self._parse_header(header_line=lines["header"][0])

        # Expect two values per line.
        positions = []
        values = []

        for line in lines["data"]:
            split = line.strip("\n").split(" ")
            if not len(split) == 2:
                raise VariableStepParseError("VariableStep block items must contain two values per line")
            try:
                position = int(split[0])
            except ValueError as e:
                raise ValueError(f"Error parsing position {split[0]}") from e

            positions.append(position)

            try:
                value = float(split[0])
            except ValueError:
                try:
                    value = int(split[0])
                except ValueError as e:
                    raise ValueError(f"Error parsing value {split[1]}") from e

            values.append(value)

        return VariableStepBlock(
            chrom=parsed_header.chrom,
            positions=positions,
            values=values,
            span=parsed_header.span,
        )

    def _parse_fixedstep_block(self, lines: Dict[str, List[str]]) -> FixedStepBlock:
        parsed_header = self._parse_header(header_line=lines["header"][0])

        values = []
        # Expect a single value per line.
        for line in lines["data"]:
            split = line.strip("\n").split(" ")
            if not len(split) != 1:
                raise FixedStepParseError("FixedStep block items must contain a single value per line")

            try:
                value = float(split[0])
            except ValueError:
                try:
                    value = int(split[0])
                except ValueError as e:
                    raise ValueError(f"Error parsing value {split[0]}") from e

            values.append(value)

        return FixedStepBlock(
            chrom=parsed_header.chrom,
            start=parsed_header.start,
            step=parsed_header.step,
            values=values,
            span=parsed_header.span,
        )


# IDEA: we want to have multiple sets of intervals (each with their own header)
# so as to simplify the export: we can verify that there's no overlap,
# we can verify that all intervals are of the same type (if required), etc.
# class CollectionWiggleBlocks:
#    def __init__(self, wiggle_blocks=List[WiggleBlock]):
#        self._blocks = sorted(wiggle_blocks)  # TODO:
#        self._validate_blocks()
#
#    def _validate_blocks(self):
#        # Verify that the intervals are sorted, non-overlapping,
#        # and share the same properties (?) TODO: verify if required.
#        raise NotImplementedError
#
#    def add_block(self, interval: WiggleBlock):
#        self._intervals.append(interval)
#        self._intervals = sorted(self._intervals)
#        self._validate_blocks()
#
#
# def merge_variable_step_blocks(*intervals: VariableStepBlock) -> VariableStepBlock:
#    # Verify that the contained positions are non-overlapping.
#    raise NotImplementedError
#    return None
#
#
# def merge_fixed_step_blocks(*intervals: FixedStepBlock) -> FixedStepBlock:
#    # Verify that the intervals are contiguous.
#    # First : sort
#    # Then : verify that the start of the next interval is the end of the previous one.
#    raise NotImplementedError
#
#    for i in range(1, len(intervals)):
#        if intervals[i].start != intervals[i - 1].start + intervals[i - 1].span:
#            raise ValueError("FixedStep intervals must be contiguous. Consider using VariableStep instead.")
#
