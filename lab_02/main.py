import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from tkinter import *

from algs import (
		Model,
		UniformTimeGenerator,
		RayleighTimeGenerator,
		ExponentialTimeGenerator,
		WeibullTimeGenerator,
		NormalTimeGenerator,
		RequestGenerator,
		RequestProcessor,
		TimeGenerator,
)

GEN_TIME_GENERATOR_TYPE = NormalTimeGenerator
PROC_TIME_GENERATOR_TYPE = NormalTimeGenerator

ONE_PARAM_GENERATOR_TYPES = [
		RayleighTimeGenerator,
		ExponentialTimeGenerator,
		WeibullTimeGenerator,
]
TWO_PARAMS_GENERATOR_TYPES = [UniformTimeGenerator, NormalTimeGenerator]

XMATRIX_CELL_WIDTH = 7
YMATRIX_CELL_WIDTH = 6
NUMBER_CELL_WIDTH = 2
ENTRY_CELL_WIDTH = 4

LABEL_CELL_COLOR = "gray87"
COMMON_CELL_COLOR = "white"

EQUATION_LABELS = [
		"Линейное нормированное: ",
		"Нелинейное нормированное: ",
		"Линейное натуральное: ",
		"Нелинейное натуральное: ",
]

EQUATION_X_FOREGROUND = "black"
EQUATION_COEF_FOREGROUND = "blue"
EQUATION_X_BACKGROUND = "white"
EQUATION_COEF_BACKGROUND = "white"
EQUATION_FONT = ("Courier", 15, "bold")


def countFactorsByGeneratorType(genType):
		if genType in ONE_PARAM_GENERATOR_TYPES:
				return 1
		elif genType in TWO_PARAMS_GENERATOR_TYPES:
				return 2


@dataclass
class Combination:
		elems: tuple[int]
		value: any

		def __init__(self, elems, value):
				self.elems = elems
				self.value = value


@dataclass
class Interval:
		min: float
		max: float

		def __init__(self, min, max):
				self.min = min
				self.max = max


@dataclass
class YMatrix:
		y: list[float]
		yLinear: list[float]
		yNonLinear: list[float]
		deltaLinear: list[float]
		deltaNonLinear: list[float]

		labels: list[str]
		factorsCount: int
		colsCount: int
		rowsCount: int

		rowMode: bool

		def __init__(self, factorsCount: int, rowMode: bool = False):
				self.y = self.yLinear = self.yNonLinear = self.deltaLinear = self.deltaNonLinear = []
				self.factorsCount = factorsCount

				self.labels = ["y", "y_lin", "y_nlin", "Δy_lin", "Δy_nlin"]
				self.colsCount = len(self.labels)

				self.rowMode = rowMode
				if rowMode:
						self.rowsCount = 1
				else:
						self.rowsCount = 2**factorsCount

		def calculateY(
				self,
				factorMatrix: list[list[int]],
				time: int,
				genTimeGeneratorType: any,
				procTimeGeneratorType: any,
		):
				self.y = []
				for factors in factorMatrix:
						genTimeGenerator, procTimeGenerator = self.__initTimeGenerators(
								factors, genTimeGeneratorType, procTimeGeneratorType
						)
						processor = RequestProcessor(procTimeGenerator)
						generator = RequestGenerator(genTimeGenerator, [processor])

						model = Model([generator], [processor])
						result = model.simulateEventBased(time)
						# lambdaReal = 1 / (result.generators[0].totalGenerationTime / result.generators[0].totalRequests)
						# muReal = 1 / (result.processors[0].totalProcessingTime / result.processors[0].totalRequests)

						# systemLoadReal = lambdaReal / muReal
						# self.y.append(systemLoadReal)
						self.y.append(result.processors[0].avgWaitingTime)


		def __solveEquation(self, coefs: list[float], x: list[float]):
				y = 0
				for i, val in enumerate(x):
						y += coefs[i] * val
				return y

		def calculateYLinear(self, linearEqCoefs: list[float], xMatrixMain: list[list[int]]):
				self.yLinear = []
				for i in range(len(xMatrixMain[0])):
						self.yLinear.append(
								self.__solveEquation(linearEqCoefs, [xMatrixMain[j][i] for j in range(len(xMatrixMain))])
						)

		def calculateYNonLinear(self, nonLinearEqCoefs: list[float], xMatrixFull: list[list[int]]):
				self.yNonLinear = []
				for i in range(len(xMatrixFull[0])):
						self.yNonLinear.append(
								self.__solveEquation(
										nonLinearEqCoefs,
										[xMatrixFull[j][i] for j in range(len(xMatrixFull))],
								)
						)

		def calculateDeltaLinear(self):
				if not (self.y and self.yLinear):
						return
				self.deltaLinear = []
				for i in range(len(self.y)):
						self.deltaLinear.append(abs(self.y[i] - self.yLinear[i]))

		def calculateDeltaNonLinear(self):
				if not (self.y and self.yNonLinear):
						return
				self.deltaNonLinear = []
				for i in range(len(self.y)):
						self.deltaNonLinear.append(abs(self.y[i] - self.yNonLinear[i]))

		def __initTimeGenerators(self, factors: list[int], genTimeGeneratorType: any, procTimeGeneratorType: any):
				genFactorsCount = countFactorsByGeneratorType(genTimeGeneratorType)
				procFactorsCount = countFactorsByGeneratorType(procTimeGeneratorType)
				if genFactorsCount == 1 and procFactorsCount == 1:
						return genTimeGeneratorType(factors[0]), procTimeGeneratorType(factors[1])
				elif genFactorsCount == 1 and procFactorsCount == 2:
						return genTimeGeneratorType(factors[0]), procTimeGeneratorType(factors[1], factors[2])
				elif genFactorsCount == 2 and procFactorsCount == 1:
						return genTimeGeneratorType(factors[0], factors[2]), procTimeGeneratorType(factors[1])
				elif genFactorsCount == 2 and procFactorsCount == 2:
						return genTimeGeneratorType(factors[0], factors[2]), procTimeGeneratorType(factors[1], factors[3])

		def full(self):
				return [
						self.y,
						self.yLinear,
						self.yNonLinear,
						self.deltaLinear,
						self.deltaNonLinear,
				]


@dataclass
class XMatrix:
		# список столбцов x_i
		main: list[list[int]]
		# список столбцов-комбинаций (x1x2, x2x3, ...)
		combs: list[Combination]

		_full: list[list[int]]

		# список названий столбцов
		labels: list[str]
		# кол-во факторов (2, 3 или 4)
		factorsCount: int

		colsCount: int
		rowsCount: int

		rowMode: bool

		def __init__(self, factorsCount: int, rowMode: bool = False):
				self.factorsCount = factorsCount
				self.colsCount = 2**factorsCount
				if rowMode:
						self.rowsCount = 1
				else:
						self.rowsCount = 2**factorsCount

				self.rowMode = rowMode
				# if not rowMode:
				self.__initMain()
				self.__initCombs()
				self.__initFull()

				self.__initLabels()

		def __initMain(self):
				self.main = []
				if self.rowMode:
						for i in range(self.factorsCount + 1):
								if i == 0:
										val = [1]
								else:
										val = [None]
								self.main.append(val)
				else:
						prod = list(product([-1, 1], repeat=self.factorsCount))
						for i in range(self.factorsCount + 1):
								if i == 0:
										val = [1 for j in range(2**self.factorsCount)]
								else:
										val = [p[i - 1] for p in prod]
								self.main.append(val)

		def __initCombs(self):
				self.combs = []
				idx = [i + 1 for i in range(self.factorsCount)]
				combs = list(combinations(idx, 2)) + list(combinations(idx, 3)) + list(combinations(idx, 4))
				for comb in combs:
						val = []
						for elem in comb:
								val = multTwoLists(val, self.main[elem])
						self.combs.append(Combination(comb, val))

		def __initLabels(self):
				idx = [i + 1 for i in range(self.factorsCount)]
				combs = list(combinations(idx, 2)) + list(combinations(idx, 3)) + list(combinations(idx, 4))

				self.labels = ["x" + str(i) for i in range(self.factorsCount + 1)]
				for comb in combs:
						label = ""
						for elem in comb:
								label += "x" + str(elem)
						self.labels.append(label)

		def __initFull(self):
				self._full = []
				for col in self.main:
						self._full.append(col)
				for comb in self.combs:
						self._full.append(comb.value)

		def setRowFactors(self, normFactors: list[float]):
				self.main = []
				for i in range(self.factorsCount + 1):
						if i == 0:
								val = [1]
						else:
								val = [f[i - 1] for f in normFactors]
						self.main.append(val)
				self.__initCombs()
				self.__initFull()

		# вся матрица целиком (объединение одиночных столбцов и столбцов-комбинаций)
		def full(self):
				return self._full

		# факторная матрица
		# содержит значения x_i (без нулевого и без комбинаций) по строкам
		def factor(self):
				colsMtrx = self.main[1:]
				rowsMtrx = []
				for i in range(len(colsMtrx[0])):
						row = []
						for j in range(len(colsMtrx)):
								row.append(colsMtrx[j][i])
						rowsMtrx.append(row)
				return rowsMtrx


def multTwoLists(list1: list[int], list2: list[int]):
		if len(list1) == 0:
				return list2
		if len(list2) == 0:
				return list1
		return [elem1 * elem2 if elem1 != None and elem2 != None else None for elem1, elem2 in zip(list1, list2)]


def subtractTwoLists(list1: list[int], list2: list[int]):
		if len(list1) == 0:
				return list2
		if len(list2) == 0:
				return list1
		return [elem1 - elem2 if elem1 != None and elem2 != None else None for elem1, elem2 in zip(list1, list2)]


@dataclass
class PlanningMatrix:
		frame: Frame
		column: int
		row: int
		columnspan: int
		rowspan: int
		padx: int
		pady: int
		window: any

		factorsCount: int
		rowsCount: int

		xMatrix: XMatrix
		yMatrix: YMatrix
		normFactorMatrix: list[list[int]]
		naturFactorMatrix: list[list[float]]

		normLinearCoefs: list[float]
		normNonLinearCoefs: list[float]

		intervals: list[Interval]
		points: list[float]

		xFields: list[Entry]
		yFields: list[Entry]

		genTimeGeneratorType: TimeGenerator
		procTimeGeneratorType: TimeGenerator

		rowMode: bool

		def __init__(
				self,
				genTimeGeneratorType: TimeGenerator,
				procTimeGeneratorType: TimeGenerator,
				rowMode: bool = False,
		):
				self.factorsCount = countFactorsByGeneratorType(genTimeGeneratorType) + countFactorsByGeneratorType(
						procTimeGeneratorType
				)
				self.xColsCount = 2**self.factorsCount
				self.yColsCount = 5

				self.rowMode = rowMode
				if rowMode:
						self.rowsCount = 1
				else:
						self.rowsCount = 2**self.factorsCount + 1

				self.genTimeGeneratorType = genTimeGeneratorType
				self.procTimeGeneratorType = procTimeGeneratorType

				self.xMatrix = XMatrix(factorsCount=self.factorsCount, rowMode=rowMode)
				self.yMatrix = YMatrix(factorsCount=self.factorsCount, rowMode=rowMode)

		def grid(self, window, column=0, row=0, columnspan=1, rowspan=1, padx=0, pady=0):
				self.window = window
				self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = (
						column,
						row,
						columnspan,
						rowspan,
						padx,
						pady,
				)

				self.frame = Frame(window)
				self.frame.grid(
						row=row,
						column=column,
						rowspan=rowspan,
						columnspan=columnspan,
						sticky=NS,
						padx=10,
						pady=10,
				)
				self.frame.grid_rowconfigure(0, weight=1)
				self.frame.grid_columnconfigure(0, weight=1)

				self.__createXLabels()
				self.__createXFields()

				self.__createYLabels()
				self.__createYFields()

		def __createCell(self, row: int, column: int, width: int, color: str, value=None):
				cell = Entry(self.frame, highlightthickness=1, relief=FLAT, justify=CENTER, width=width)
				if value != None:
						cell.insert(0, str(value))
				cell.config(state="readonly", readonlybackground=color)
				cell.grid(row=row, column=column, sticky=NSEW, padx=0, pady=0)
				return cell

		def __createXLabels(self):
				for j, label in enumerate(self.xMatrix.labels):
						# для ширины колонок по длине содержимого (x-колонки):
						colWidth = len(label) + 1
						# для фиксированной ширины колонок (x-колонки):
						# colWidth = XMATRIX_CELL_WIDTH
						self.__createCell(row=0, column=j + 1, width=colWidth, color=LABEL_CELL_COLOR, value=label)

				for i in range(self.xMatrix.rowsCount + 1):
						value = None
						if i != 0:
								value = i
						self.__createCell(
								row=i,
								column=0,
								width=NUMBER_CELL_WIDTH,
								color=LABEL_CELL_COLOR,
								value=value,
						)

		def __destroyXFields(self):
				for i in range(len(self.xFields)):
						for j in range(len(self.xFields[0])):
								self.xFields[i][j].destroy()

		def __clearXFields(self):
				for i in range(len(self.xFields)):
						for j in range(len(self.xFields[0])):
								print(f"{i=}, {j=}")
								self.xFields[i][j].config(state="normal", readonlybackground=COMMON_CELL_COLOR)
								self.xFields[i][j].delete(0, END)
								self.xFields[i][j].config(state="readonly", readonlybackground=COMMON_CELL_COLOR)

		def __createXFields(self):
				self.xFields = []
				for j in range(self.xMatrix.colsCount):
						col = []
						colWidth = len(self.xMatrix.labels[j])
						for i in range(self.xMatrix.rowsCount):
								cell = self.__createCell(
										row=i + 1,
										column=j + 1,
										width=colWidth,
										color=COMMON_CELL_COLOR,
										value=self.xMatrix.full()[j][i],
								)
								col.append(cell)
						self.xFields.append(col)

		def __updateXFields(self):
				for j in range(self.xMatrix.colsCount):
						for i in range(self.xMatrix.rowsCount):
								value = self.xMatrix.full()[j][i]
								self.xFields[j][i].config(state="normal", readonlybackground=COMMON_CELL_COLOR)
								self.xFields[j][i].delete(0, END)
								if self.rowMode:
										value = round(value, 2)
								self.xFields[j][i].insert(0, value)
								self.xFields[j][i].config(state="readonly", readonlybackground=COMMON_CELL_COLOR)

		def __destroyYFields(self):
				for i in range(len(self.yFields)):
						for j in range(len(self.yFields[0])):
								self.yFields[i][j].destroy()

		def __clearYFields(self):
				for i in range(len(self.yFields)):
						for j in range(len(self.yFields[0])):
								self.yFields[i][j].config(state="normal", readonlybackground=COMMON_CELL_COLOR)
								self.yFields[i][j].delete(0, END)
								self.yFields[i][j].config(state="readonly", readonlybackground=COMMON_CELL_COLOR)

		def __createYLabels(self):
				offset = self.xMatrix.colsCount
				for j, label in enumerate(self.yMatrix.labels):
						# для ширины колонок по длине содержимого (y-колонки):
						# colWidth = len(label)
						# для фиксированной ширины колонок (y-колонки):
						colWidth = YMATRIX_CELL_WIDTH
						self.__createCell(
								row=0,
								column=j + 1 + offset,
								width=colWidth,
								color=LABEL_CELL_COLOR,
								value=label,
						)

		def __createYFields(self):
				offset = self.xMatrix.colsCount
				self.yFields = []
				for j in range(self.yMatrix.colsCount):
						col = []
						colWidth = len(self.yMatrix.labels[j])
						for i in range(self.yMatrix.rowsCount):
								value = None
								if len(self.yMatrix.full()[j]):
										value = round(self.yMatrix.full()[j][i], 4)
								cell = self.__createCell(
										row=i + 1,
										column=j + 1 + offset,
										width=colWidth,
										color=COMMON_CELL_COLOR,
										value=value,
								)
								col.append(cell)
						self.yFields.append(col)

		def __updateYFields(self):
				for j in range(self.yMatrix.colsCount):
						for i in range(self.yMatrix.rowsCount):
								value = None
								if len(self.yMatrix.full()[j]):
										value = round(self.yMatrix.full()[j][i], 4)
										self.yFields[j][i].config(state="normal", readonlybackground=COMMON_CELL_COLOR)
										self.yFields[j][i].delete(0, END)
										self.yFields[j][i].insert(0, value)
										self.yFields[j][i].config(state="readonly", readonlybackground=COMMON_CELL_COLOR)

		def __makeNormalFactorMatrixNatural(self, normFactorMatrix):
				naturFactorMatrix = []
				for i in range(len(normFactorMatrix)):
						row = []
						for j in range(len(normFactorMatrix[0])):
								normValue = normFactorMatrix[i][j]
								if normValue == -1:
										naturValue = self.intervals[j].min
								elif normValue == 1:
										naturValue = self.intervals[j].max
								row.append(naturValue)
						naturFactorMatrix.append(row)
				return naturFactorMatrix

		def __makeNaturalFactorMatrixNormal(self, naturFactorMatrix):
				normFactorMatrix = []
				for i in range(len(naturFactorMatrix)):
						row = []
						for j in range(len(naturFactorMatrix[0])):
								naturValue = naturFactorMatrix[i][j]
								left = self.intervals[j].min
								right = self.intervals[j].max
								length = right - left
								middle = left + length / 2
								normValue = (naturValue - middle) / (length / 2)
								row.append(normValue)
						normFactorMatrix.append(row)
				return normFactorMatrix

		def setFactors(self, intervals: list[Interval], points: list[float] = None):
				if len(intervals) != self.factorsCount:
						raise (
								"Incorrect number of intervals. Factors number = ",
								self.factorsCount,
						)
				self.intervals = intervals
				if self.rowMode:
						if points == None:
								raise ("Points values not found. Pass them or turn the rowMode off")
						if len(points) != self.factorsCount:
								raise (
										"Incorrect number of points. Factors number = ",
										self.factorsCount,
								)

						self.points = points
						self.naturFactorMatrix = [points]

						self.normFactorMatrix = self.__makeNaturalFactorMatrixNormal(self.naturFactorMatrix)
						self.xMatrix.setRowFactors(self.normFactorMatrix)
						self.__updateXFields()

				if not self.rowMode:
						self.normFactorMatrix = self.xMatrix.factor()
						self.naturFactorMatrix = self.__makeNormalFactorMatrixNatural(self.normFactorMatrix)

		def calculateNormNonLinearCoefs(self) -> list[float]:
				coefs = []
				for j in range(self.xMatrix.colsCount):
						coef = 0
						for i in range(self.xMatrix.rowsCount):
								coef += self.xMatrix.full()[j][i] * self.yMatrix.y[i]
						coefs.append(coef / self.xMatrix.rowsCount)
				return coefs

		def normalCoefsToNaturalLinear2(self, an: Sequence[float]) -> tuple[float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 3)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 3)]]

				return (
						an[0] - (an[1] * xc[1] / dx[1]) - (an[2] * xc[2] / dx[2]),
						(an[1] / dx[1]),
						(an[2] / dx[2]),
				)

		def normalCoefsToNaturalLinear3(self, an: Sequence[float]) -> tuple[float, float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 4)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 4)]]

				return (
						an[0] - xc[1] * (an[1] / dx[1]) - xc[2] * (an[2] / dx[2]) - xc[3] * (an[3] / dx[3]),
						(an[1] / dx[1]),
						(an[2] / dx[2]),
						(an[3] / dx[3]),
				)

		def normalCoefsToNaturalLinear4(self, an: Sequence[float]) -> tuple[float, float, float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 5)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 5)]]

				return (
						an[0]
						- xc[1] * (an[1] / dx[1])
						- xc[2] * (an[2] / dx[2])
						- xc[3] * (an[3] / dx[3])
						- xc[4] * (an[4] / dx[4]),
						(an[1] / dx[1]),
						(an[2] / dx[2]),
						(an[3] / dx[3]),
						(an[4] / dx[4]),
				)

		def normalCoefsToNaturalNonLinear2(self, an: Sequence[float]) -> tuple[float, float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 3)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 3)]]

				def dx_(i: int, j: int) -> float:
						return dx[i] * dx[j]

				return (
						an[0] - (an[1] * xc[1] / dx[1]) - (an[2] * xc[2] / dx[2]) + (an[3] * xc[1] * xc[2] / dx_(1, 2)),
						(an[1] / dx[1]) - (an[3] * xc[2] / dx_(1, 2)),
						(an[2] / dx[2]) - (an[3] * xc[1] / dx_(1, 2)),
						(an[3] / dx_(1, 2)),
				)

		def normalCoefsToNaturalNonLinear3(
				self, an: Sequence[float]
		) -> tuple[float, float, float, float, float, float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 4)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 4)]]

				# helper for x_i(0) / delta_x_i
				def k(i: int) -> float:
						return xc[i] / dx[i]

				return (
						(
								an[0]
								- an[1] * k(1)
								- an[2] * k(2)
								- an[3] * k(3)
								+ an[4] * k(1) * k(2)
								+ an[5] * k(1) * k(3)
								+ an[6] * k(2) * k(3)
								- an[7] * k(1) * k(2) * k(3)
						),
						(an[1] - an[4] * k(2) - an[5] * k(3) + an[7] * k(2) * k(3)) / dx[1],
						(an[2] - an[4] * k(1) - an[6] * k(3) + an[7] * k(1) * k(3)) / dx[2],
						(an[3] - an[5] * k(1) - an[6] * k(2) + an[7] * k(1) * k(2)) / dx[3],
						(an[4] - an[7] * k(3)) / dx[1] / dx[2],
						(an[5] - an[7] * k(2)) / dx[1] / dx[3],
						(an[6] - an[7] * k(1)) / dx[2] / dx[3],
						an[7] / dx[1] / dx[2] / dx[3],
				)

		def normalCoefsToNaturalNonLinear4(
				self, an: Sequence[float]
		) -> tuple[
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
				float,
		]:
				# intervals = self.intervals
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 5)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 5)]]

				# helper for x_i(0) / delta_x_i
				def k(i: int) -> float:
						return xc[i] / dx[i]

				return (
						(
								an[0]
								- an[1] * k(1)
								- an[2] * k(2)
								- an[3] * k(3)
								- an[4] * k(4)
								+ an[5] * k(1) * k(2)
								+ an[6] * k(1) * k(3)
								+ an[7] * k(1) * k(4)
								+ an[8] * k(2) * k(3)
								+ an[9] * k(2) * k(4)
								+ an[10] * k(3) * k(4)
								- an[11] * k(1) * k(2) * k(3)
								- an[12] * k(1) * k(2) * k(4)
								- an[13] * k(1) * k(3) * k(4)
								- an[14] * k(2) * k(3) * k(4)
								+ an[15] * k(1) * k(2) * k(3) * k(4)
						),
						(
								an[1]
								- an[5] * k(2)
								- an[6] * k(3)
								- an[7] * k(4)
								+ an[11] * k(2) * k(3)
								+ an[12] * k(2) * k(4)
								+ an[13] * k(3) * k(4)
								- an[15] * k(2) * k(3) * k(4)
						)
						/ dx[1],
						(
								an[2]
								- an[5] * k(1)
								- an[8] * k(3)
								- an[9] * k(4)
								+ an[11] * k(1) * k(3)
								+ an[12] * k(1) * k(4)
								+ an[14] * k(3) * k(4)
								- an[15] * k(1) * k(3) * k(4)
						)
						/ dx[2],
						(
								an[3]
								- an[6] * k(1)
								- an[8] * k(2)
								- an[10] * k(4)
								+ an[11] * k(1) * k(2)
								+ an[13] * k(1) * k(4)
								+ an[14] * k(2) * k(4)
								- an[15] * k(1) * k(2) * k(4)
						)
						/ dx[3],
						(
								an[4]
								- an[7] * k(1)
								- an[9] * k(2)
								- an[10] * k(3)
								+ an[12] * k(1) * k(2)
								+ an[13] * k(1) * k(3)
								+ an[14] * k(2) * k(3)
								- an[15] * k(1) * k(2) * k(3)
						)
						/ dx[4],
						(an[5] - an[11] * k(3) - an[12] * k(4) + an[15] * k(3) * k(4)) / dx[1] / dx[2],
						(an[6] - an[11] * k(2) - an[13] * k(4) + an[15] * k(2) * k(4)) / dx[1] / dx[3],
						(an[7] - an[12] * k(2) - an[13] * k(3) + an[15] * k(2) * k(3)) / dx[1] / dx[4],
						(an[8] - an[11] * k(1) - an[14] * k(4) + an[15] * k(1) * k(4)) / dx[2] / dx[3],
						(an[9] - an[12] * k(1) - an[14] * k(3) + an[15] * k(1) * k(3)) / dx[2] / dx[4],
						(an[10] - an[13] * k(1) - an[14] * k(2) + an[15] * k(1) * k(2)) / dx[3] / dx[4],
						(an[11] - an[15] * k(4)) / dx[1] / dx[2] / dx[3],
						(an[12] - an[15] * k(3)) / dx[1] / dx[2] / dx[4],
						(an[13] - an[15] * k(2)) / dx[1] / dx[3] / dx[4],
						(an[14] - an[15] * k(1)) / dx[2] / dx[3] / dx[4],
						an[15] / dx[1] / dx[2] / dx[3] / dx[4],
				)

		def normalCoefsToNaturalLinear4(self, an: Sequence[float]) -> tuple[float, float, float, float, float]:
				X_MIN = (None, *[interval.min for interval in self.intervals])
				X_MAX = (None, *[interval.max for interval in self.intervals])

				dx = [None, *[(X_MAX[i] - X_MIN[i]) / 2 for i in range(1, 5)]]
				xc = [None, *[(X_MAX[i] + X_MIN[i]) / 2 for i in range(1, 5)]]

				return (
						an[0]
						- xc[1] * (an[1] / dx[1])
						- xc[2] * (an[2] / dx[2])
						- xc[3] * (an[3] / dx[3])
						- xc[4] * (an[4] / dx[4]),
						(an[1] / dx[1]),
						(an[2] / dx[2]),
						(an[3] / dx[3]),
						(an[4] / dx[4]),
				)

		def calculate(self):
				self.yMatrix.calculateY(
						factorMatrix=self.naturFactorMatrix,
						time=100,
						genTimeGeneratorType=self.genTimeGeneratorType,
						procTimeGeneratorType=self.procTimeGeneratorType,
				)

				self.normNonLinearCoefs = self.calculateNormNonLinearCoefs()
				self.normLinearCoefs = self.normNonLinearCoefs[: (self.factorsCount + 1)]

				self.yMatrix.calculateYLinear(self.normLinearCoefs, self.xMatrix.main)
				self.yMatrix.calculateYNonLinear(self.normNonLinearCoefs, self.xMatrix.full())
				self.yMatrix.calculateDeltaLinear()
				self.yMatrix.calculateDeltaNonLinear()

				self.__updateYFields()
				self.__updateXFields()

		def __getStr(self, value):
				return str(round(value, 3) or round(value, 4) or 0.0)
				# string = str(round(value, 2))
				# if string == "0.0":
				# 	return "0"
				# else:
				# 	return string

		def __getEquation(self, coefs):
				equation = "y = "
				for i, coef in enumerate(coefs):
						sign = "+" if coef >= 0 else "-"
						if i != 0:
								equation += sign
						equation += self.__getStr(abs(coef)) + self.xMatrix.labels[i]
				return equation

		def getNormalLinearEquation(self):
				return self.__getEquation(self.normLinearCoefs)

		def getNormalNonLinearEquation(self):
				return self.__getEquation(self.normNonLinearCoefs)

		def getNaturalNonLinearEquation(self):
				if (
						self.genTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
						and self.procTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
				):
						naturalNonLinearCoefs = self.normalCoefsToNaturalNonLinear2(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
						and self.procTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
				):
						naturalNonLinearCoefs = self.normalCoefsToNaturalNonLinear3(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
						and self.procTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
				):
						naturalNonLinearCoefs = self.normalCoefsToNaturalNonLinear3(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
						and self.procTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
				):
						naturalNonLinearCoefs = self.normalCoefsToNaturalNonLinear4(self.normNonLinearCoefs)
				print(f"{naturalNonLinearCoefs=}")
				return self.__getEquation(naturalNonLinearCoefs)

		def getNaturalLinearEquation(self):
				if (
						self.genTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
						and self.procTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
				):
						naturalLinearCoefs = self.normalCoefsToNaturalLinear2(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
						and self.procTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
				):
						naturalLinearCoefs = self.normalCoefsToNaturalLinear3(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
						and self.procTimeGeneratorType in ONE_PARAM_GENERATOR_TYPES
				):
						naturalLinearCoefs = self.normalCoefsToNaturalLinear3(self.normNonLinearCoefs)
				if (
						self.genTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
						and self.procTimeGeneratorType in TWO_PARAMS_GENERATOR_TYPES
				):
						naturalLinearCoefs = self.normalCoefsToNaturalLinear4(self.normNonLinearCoefs)
				print(f"{naturalLinearCoefs=}")
				return self.__getEquation(naturalLinearCoefs)


@dataclass
class IntervalDataBlock:
		frame: Frame
		column: int
		row: int
		columnspan: int
		rowspan: int
		padx: int
		pady: int
		window: any

		genTimeGeneratorType: any
		procTimeGeneratorType: any

		genFactorsCount: int
		procFactorsCount: int

		genIntensityField: tuple[Entry]
		genRangeField: tuple[Entry]
		procIntensityField: tuple[Entry]
		prcoRangeField: tuple[Entry]

		def __init__(self, genTimeGeneratorType: any, procGeneratorType: any):
				self.genTimeGeneratorType = genTimeGeneratorType
				self.procGeneratorType = procGeneratorType

				self.genFactorsCount = countFactorsByGeneratorType(genTimeGeneratorType)
				self.procFactorsCount = countFactorsByGeneratorType(procGeneratorType)

		def grid(self, window, row=0, column=0, columnspan=1, rowspan=1, padx=0, pady=0):
				self.window = window
				self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = (
						column,
						row,
						columnspan,
						rowspan,
						padx,
						pady,
				)

				self.frame = self.__createFrame(
						window,
						row=row,
						column=column,
						rowspan=rowspan,
						columnspan=columnspan,
						sticky=NSEW,
						text="Интервал:",
				)

				self.genIntensityField = self.genRangeField = self.procIntensityField = self.procRangeField = None

				timeGeneratorsTypes = [self.genTimeGeneratorType, self.procGeneratorType]
				for i, type in enumerate(timeGeneratorsTypes):
						if i == 0:
								frameText = "Поступление заявок:"
						elif i == 1:
								frameText = "Обработка заявок:"
						frame = self.__createFrame(self.frame, row=1, column=i, sticky=NSEW, text=frameText)

						label = self.__getTimeGeneratorLabel(type)

						lblDistribution = Label(frame, text=label, font=("", 13, "italic"))
						lblDistribution.grid(row=0, column=i, columnspan=3, sticky=EW)

						lblIntensity = Label(frame, text="Интенсивность:")
						lblIntensity.grid(row=2, column=i, sticky=W)

						lblIntensityMin = Label(frame, text="min:")
						lblIntensityMin.grid(row=1, column=i + 1, sticky=EW)

						lblIntensityMax = Label(frame, text="max:")
						lblIntensityMax.grid(row=1, column=i + 2, sticky=EW)

						txtIntensityMin = self.__createEntry(frame, row=2, column=i + 1, sticky=E)
						txtIntensityMax = self.__createEntry(frame, row=2, column=i + 2, sticky=W)
						if i == 0:
								self.genIntensityField = (txtIntensityMin, txtIntensityMax)
						elif i == 1:
								self.procIntensityField = (txtIntensityMin, txtIntensityMax)

						if countFactorsByGeneratorType(type) == 2:
								lblRange = Label(frame, text="Разброс интенсивности:")
								lblRange.grid(row=3, column=i, sticky=W)

								txtRangeMin = self.__createEntry(frame, row=3, column=i + 1, sticky=E)
								txtRangeMax = self.__createEntry(frame, row=3, column=i + 2, sticky=W)
								if i == 0:
										self.genRangeField = (txtRangeMin, txtRangeMax)
								elif i == 1:
										self.procRangeField = (txtRangeMin, txtRangeMax)

		def __createEntry(self, window, row=0, column=0, sticky=NSEW, value=None):
				entry = Entry(window, width=ENTRY_CELL_WIDTH)
				if value != None:
						entry.insert(0, str(value))
				entry.grid(row=row, column=column, sticky=sticky)
				return entry

		def __createFrame(self, window, row=0, column=0, columnspan=1, rowspan=1, sticky=NSEW, text=None):
				if text != None:
						frame = LabelFrame(window, text=text)
				else:
						frame = Frame(window)

				frame.grid(
						row=row,
						column=column,
						columnspan=columnspan,
						rowspan=rowspan,
						sticky=sticky,
						padx=10,
						pady=10,
				)
				frame.grid_rowconfigure(0, weight=1)
				frame.grid_columnconfigure(0, weight=1)

				return frame

		def __getTimeGeneratorLabel(self, timeGeneratorType: any):
				if timeGeneratorType is NormalTimeGenerator:
						return "Нормальный закон"
				elif timeGeneratorType is ExponentialTimeGenerator:
						return "Экспоненциальный закон"
				elif timeGeneratorType is UniformTimeGenerator:
						return "Равномерный закон"
				elif timeGeneratorType is WeibullTimeGenerator:
						return "Закон Вейбулла"
				elif timeGeneratorType is RayleighTimeGenerator:
						return "Закон Рэлея"

		def factors(self):
				factors = []
				for field in [
						self.genIntensityField,
						self.procIntensityField,
						self.genRangeField,
						self.procRangeField,
				]:
						if field != None:
								if field[0].get() != "" and field[1].get() != "":
										factors.append(Interval(float(field[0].get()), float(field[1].get())))
								else:
										factors.append(None)
				return factors

		def setGenFactors(self, intensity: Interval, range: Interval = None):
				self.genIntensityField[0].delete(0, END)
				self.genIntensityField[1].delete(0, END)
				self.genIntensityField[0].insert(0, str(intensity.min))
				self.genIntensityField[1].insert(0, str(intensity.max))
				if self.genRangeField != None:
						self.genRangeField[0].delete(0, END)
						self.genRangeField[1].delete(0, END)
						self.genRangeField[0].insert(0, str(range.min))
						self.genRangeField[1].insert(0, str(range.max))

		def setProcFactors(self, intensity: Interval, range: Interval = None):
				self.procIntensityField[0].delete(0, END)
				self.procIntensityField[1].delete(0, END)
				self.procIntensityField[0].insert(0, str(intensity.min))
				self.procIntensityField[1].insert(0, str(intensity.max))
				if self.procRangeField != None:
						self.procRangeField[0].delete(0, END)
						self.procRangeField[1].delete(0, END)
						self.procRangeField[0].insert(0, str(range.min))
						self.procRangeField[1].insert(0, str(range.max))


@dataclass
class PointDataBlock:
		frame: Frame
		column: int
		row: int
		columnspan: int
		rowspan: int
		padx: int
		pady: int
		window: any

		genTimeGeneratorType: any
		procTimeGeneratorType: any

		genFactorsCount: int
		procFactorsCount: int

		genIntensityField: Entry
		genRangeField: Entry
		procIntensityField: Entry
		prcoRangeField: Entry

		def __init__(self, genTimeGeneratorType: any, procGeneratorType: any):
				self.genTimeGeneratorType = genTimeGeneratorType
				self.procGeneratorType = procGeneratorType

				self.genFactorsCount = countFactorsByGeneratorType(genTimeGeneratorType)
				self.procFactorsCount = countFactorsByGeneratorType(procGeneratorType)

		def grid(self, window, row=0, column=0, columnspan=1, rowspan=1, padx=0, pady=0):
				self.window = window
				self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = (
						column,
						row,
						columnspan,
						rowspan,
						padx,
						pady,
				)

				self.frame = self.__createFrame(
						window,
						row=row,
						column=column,
						rowspan=rowspan,
						columnspan=columnspan,
						sticky=NSEW,
						text="Точка:",
				)

				self.genIntensityField = self.genRangeField = self.procIntensityField = self.procRangeField = None

				timeGeneratorsTypes = [self.genTimeGeneratorType, self.procGeneratorType]
				for i, type in enumerate(timeGeneratorsTypes):
						if i == 0:
								frameText = "Поступление заявок:"
						elif i == 1:
								frameText = "Обработка заявок:"
						frame = self.__createFrame(self.frame, row=1, column=i, sticky=NSEW, text=frameText)

						label = self.__getTimeGeneratorLabel(type)

						lblDistribution = Label(frame, text=label, font=("", 13, "italic"))
						lblDistribution.grid(row=0, column=i, columnspan=3, sticky=EW)

						lblIntensity = Label(frame, text="Интенсивность:")
						lblIntensity.grid(row=2, column=i, sticky=W)

						lblIntensityMin = Label(frame, text="x:")
						lblIntensityMin.grid(row=1, column=i + 1, sticky=EW)

						txtIntensity = self.__createEntry(frame, row=2, column=i + 1, sticky=EW)
						if i == 0:
								self.genIntensityField = txtIntensity
						elif i == 1:
								self.procIntensityField = txtIntensity

						if countFactorsByGeneratorType(type) == 2:
								lblRange = Label(frame, text="Разброс интенсивности:")
								lblRange.grid(row=3, column=i, sticky=W)

								txtRange = self.__createEntry(frame, row=3, column=i + 1, sticky=EW)
								if i == 0:
										self.genRangeField = txtRange
								elif i == 1:
										self.procRangeField = txtRange

		def __createEntry(self, window, row=0, column=0, sticky=NSEW, value=None):
				entry = Entry(window, width=ENTRY_CELL_WIDTH)
				if value != None:
						entry.insert(0, str(value))
				entry.grid(row=row, column=column, sticky=sticky)
				return entry

		def __createFrame(self, window, row=0, column=0, columnspan=1, rowspan=1, sticky=NSEW, text=None):
				if text != None:
						frame = LabelFrame(window, text=text)
				else:
						frame = Frame(window)

				frame.grid(
						row=row,
						column=column,
						columnspan=columnspan,
						rowspan=rowspan,
						sticky=sticky,
						padx=10,
						pady=10,
				)
				frame.grid_rowconfigure(0, weight=1)
				frame.grid_columnconfigure(0, weight=1)
				return frame

		def __getTimeGeneratorLabel(self, timeGeneratorType: any):
				if timeGeneratorType is NormalTimeGenerator:
						return "Нормальный закон"
				elif timeGeneratorType is ExponentialTimeGenerator:
						return "Экспоненциальный закон"
				elif timeGeneratorType is UniformTimeGenerator:
						return "Равномерный закон"
				elif timeGeneratorType is WeibullTimeGenerator:
						return "Закон Вейбулла"
				elif timeGeneratorType is RayleighTimeGenerator:
						return "Закон Рэлея"

		def factors(self):
				factors = []
				for field in [
						self.genIntensityField,
						self.procIntensityField,
						self.genRangeField,
						self.procRangeField,
				]:
						if field != None:
								if field.get() != "" and field.get() != "":
										factors.append(float(field.get()))
								else:
										factors.append(None)
				return factors

		def setGenFactors(self, intensity: Interval, range: Interval = None):
				self.genIntensityField.delete(0, END)
				self.genIntensityField.insert(0, str(intensity))
				if self.genRangeField != None:
						self.genRangeField.delete(0, END)
						self.genRangeField.insert(0, str(range))

		def setProcFactors(self, intensity: Interval, range: Interval = None):
				self.procIntensityField.delete(0, END)
				self.procIntensityField.insert(0, str(intensity))
				if self.procRangeField != None:
						self.procRangeField.delete(0, END)
						self.procRangeField.insert(0, str(range))


def calculate(planningMatrix: PlanningMatrix):
		global intervalDataBlock, pointDataBlock

		intervals = intervalDataBlock.factors()
		if planningMatrix.rowMode:
				points = pointDataBlock.factors()
				planningMatrix.setFactors(intervals=intervals, points=points)
		else:
				planningMatrix.setFactors(intervals=intervals)

		planningMatrix.calculate()

		if not planningMatrix.rowMode:
				equations = (
						planningMatrix.getNormalLinearEquation(),
						planningMatrix.getNormalNonLinearEquation(),
						planningMatrix.getNaturalLinearEquation(),
						planningMatrix.getNaturalNonLinearEquation(),
				)
				for i, eqLabel in enumerate(EQUATION_LABELS):
						insertEquation(i + 1, eqLabel, equations[i])


def insertEquation(index: int, equationLabel: str, equation: str):
		global eqText
		startPos = len(equationLabel)

		eqText.delete("{}.{}".format(index, len(equationLabel)), "{}.end".format(index))
		eqText.insert("{}.{}".format(index, len(equationLabel)), equation)

		regexpCoef = re.compile("([-+]\d\.*\d+)|(=\s){1}(\d\.*\d+)")

		eqText.tag_add("y", f"{index}.{startPos}", f"{index}.{startPos+3}")
		eqText.tag_config(
				"y",
				font=EQUATION_FONT,
				foreground=EQUATION_X_FOREGROUND,
				background=EQUATION_X_BACKGROUND,
		)

		for mIndex, m in enumerate(regexpCoef.finditer(equation), start=1):
				tagName = f"coef{mIndex}"
				eqText.tag_add(tagName, f"{index}.{m.start()+startPos}", f"{index}.{m.end()+startPos}")
				eqText.tag_config(
						tagName,
						font=EQUATION_FONT,
						foreground=EQUATION_COEF_FOREGROUND,
						background=EQUATION_COEF_BACKGROUND,
				)

		regexpX = re.compile("x\d")

		for mIndex, m in enumerate(regexpX.finditer(equation), start=1):
				tagName = f"x{mIndex}"
				eqText.tag_add(tagName, f"{index}.{m.start()+startPos}", f"{index}.{m.end()+startPos}")
				eqText.tag_config(
						tagName,
						font=EQUATION_FONT,
						foreground=EQUATION_X_FOREGROUND,
						background=EQUATION_X_BACKGROUND,
				)


window = Tk()

window.title("Лабораторная работа №2")
window.grid_propagate()

intervalDataBlock = IntervalDataBlock(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE)
intervalDataBlock.grid(window=window, row=0, column=0, rowspan=1)

pointDataBlock = PointDataBlock(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE)
pointDataBlock.grid(window=window, row=0, column=1, rowspan=1)

genTimeGeneratorParamsCount = 1 if GEN_TIME_GENERATOR_TYPE in ONE_PARAM_GENERATOR_TYPES else 2
procTimeGeneratorParamsCount = 1 if PROC_TIME_GENERATOR_TYPE in ONE_PARAM_GENERATOR_TYPES else 2

if genTimeGeneratorParamsCount == 1 and procTimeGeneratorParamsCount == 1:
		intervals = [Interval(1, 10), Interval(15, 90)]
		points = [i.min + (i.max - i.min) / 2 for i in intervals]

		intervalDataBlock.setGenFactors(intensity=intervals[0])
		intervalDataBlock.setProcFactors(intensity=intervals[1])

		pointDataBlock.setGenFactors(points[0])
		pointDataBlock.setProcFactors(points[1])
elif genTimeGeneratorParamsCount == 1 and procTimeGeneratorParamsCount == 2:
		intervals = [Interval(1, 10), Interval(15, 90), Interval(5, 10)]
		points = [i.min + (i.max - i.min) / 2 for i in intervals]

		intervalDataBlock.setGenFactors(intensity=intervals[0])
		intervalDataBlock.setProcFactors(intensity=intervals[1], range=intervals[2])

		pointDataBlock.setGenFactors(points[0])
		pointDataBlock.setProcFactors(points[1], points[2])
elif genTimeGeneratorParamsCount == 2 and procTimeGeneratorParamsCount == 1:
		intervals = [Interval(1, 10), Interval(15, 90), Interval(5, 10)]
		points = [i.min + (i.max - i.min) / 2 for i in intervals]

		intervalDataBlock.setGenFactors(intensity=intervals[0], range=intervals[2])
		intervalDataBlock.setProcFactors(intensity=intervals[1])

		pointDataBlock.setGenFactors(points[0], points[2])
		pointDataBlock.setProcFactors(points[1])
elif genTimeGeneratorParamsCount == 2 and procTimeGeneratorParamsCount == 2:
		intervals = [Interval(1, 10), Interval(15, 90), Interval(5, 10), Interval(5, 10)]
		points = [i.min + (i.max - i.min) / 2 for i in intervals]

		intervalDataBlock.setGenFactors(intensity=intervals[0], range=intervals[2])
		intervalDataBlock.setProcFactors(intensity=intervals[1], range=intervals[3])

		pointDataBlock.setGenFactors(points[0], points[2])
		pointDataBlock.setProcFactors(points[1], points[3])

mtrx = PlanningMatrix(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE)
mtrx.grid(window=window, row=2, column=0, columnspan=2)
mtrx.setFactors(intervals=intervals)

mtrxRow = PlanningMatrix(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE, True)
mtrxRow.grid(window=window, row=3, column=0, columnspan=2)
mtrxRow.setFactors(intervals=intervals, points=points)

btnInterval = Button(window, text="Вычислить", command=lambda: calculate(mtrx))
btnInterval.grid(row=1, column=0, sticky=EW)

btnInterval = Button(window, text="Вычислить", command=lambda: calculate(mtrxRow))
btnInterval.grid(row=1, column=1, sticky=EW)

frameEquation = LabelFrame(window, text="Уравнения:")
frameEquation.grid(row=4, column=0, columnspan=2, sticky=NSEW, padx=10, pady=10)
frameEquation.grid_rowconfigure(0, weight=1)
frameEquation.grid_columnconfigure(0, weight=1)

scrollbar = Scrollbar(frameEquation, orient="horizontal")
scrollbar.grid(row=1, column=0, sticky=EW)

eqText = Text(
		frameEquation,
		height=5,
		font=("Arial", 15),
		wrap="none",
		xscrollcommand=scrollbar.set,
)
eqText.grid(row=0, column=0, sticky=NSEW)

for eqLabel in EQUATION_LABELS:
		eqText.insert("end", eqLabel + "\n")

scrollbar.config(command=eqText.xview)

calculate(mtrx)
calculate(mtrxRow)

window.mainloop()
