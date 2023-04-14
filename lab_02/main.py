from tkinter import *
import numpy as np
from algs import Model, UniformTimeGenerator, RayleighTimeGenerator, ExponentialTimeGenerator, WeibullTimeGenerator, NormalTimeGenerator, RequestGenerator, RequestProcessor
from dataclasses import dataclass
from itertools import combinations, product

GEN_TIME_GENERATOR_TYPE = ExponentialTimeGenerator
PROC_TIME_GENERATOR_TYPE = ExponentialTimeGenerator

ONE_PARAM_GENERATOR_TYPES = [RayleighTimeGenerator, ExponentialTimeGenerator, WeibullTimeGenerator]
TWO_PARAMS_GENERATOR_TYPES = [UniformTimeGenerator, NormalTimeGenerator]

XMATRIX_CELL_WIDTH = 7
YMATRIX_CELL_WIDTH = 6
NUMBER_CELL_WIDTH = 2

LABEL_CELL_COLOR = "gray87"
COMMON_CELL_COLOR = "white"

def countFactorsByGeneratorType(genType):
	if genType in ONE_PARAM_GENERATOR_TYPES: return 1
	elif genType in TWO_PARAMS_GENERATOR_TYPES: return 2


@dataclass
class Combination:
	elems: tuple[int]
	value: list[int]

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

	def __init__(self, factorsCount: int):
		self.y = self.yLinear = self.yNonLinear = self.deltaLinear = self.deltaNonLinear = []
		self.factorsCount = factorsCount

		self.labels = ['y', 'y_lin', 'y_nlin', 'Δy_lin', 'Δy_nlin']
		self.colsCount = len(self.labels)
		self.rowsCount = 2 ** factorsCount

	def calculateY(self, factorMatrix: list[list[int]], time: int, genTimeGeneratorType: any, procTimeGeneratorType: any):
		self.y = []
		for factors in factorMatrix:
			genTimeGenerator, procTimeGenerator = self.__initTimeGenerators(factors, genTimeGeneratorType, procTimeGeneratorType)
			processor = RequestProcessor(procTimeGenerator)
			generator = RequestGenerator(genTimeGenerator, [processor])

			model = Model([generator], [processor])
			result = model.simulateEventBased(time)
			lambdaReal = 1 / (result.generators[0].totalGenerationTime / result.generators[0].totalRequests)
			muReal = 1 / (result.processors[0].totalProcessingTime / result.processors[0].totalRequests)
		
			systemLoadReal = lambdaReal / muReal
			self.y.append(systemLoadReal)

	def __solveEquation(self, coefs: list[float], x: list[float]):
		y = 0
		for i, val in enumerate(x):
			y += coefs[i] * val
		return y
	
	def calculateYLinear(self, linearEqCoefs: list[float], xMatrixMain: list[list[int]]):
		self.yLinear = []
		for i in range (len(xMatrixMain[0])):
			self.yLinear.append(self.__solveEquation(linearEqCoefs, [xMatrixMain[j][i] for j in range(len(xMatrixMain))]))

	def calculateYNonLinear(self, nonLinearEqCoefs: list[float], xMatrixFull: list[list[int]]):
		self.yNonLinear = []
		for i in range (len(xMatrixFull[0])):
			self.yNonLinear.append(self.__solveEquation(nonLinearEqCoefs, [xMatrixFull[j][i] for j in range(len(xMatrixFull))]))

	def calculateDeltaLinear(self):
		if not (self.y and self.yLinear): return
		self.deltaLinear = []
		for i in range(len(self.y)):
			self.deltaLinear.append(abs(self.y[i] - self.yLinear[i]))

	def calculateDeltaNonLinear(self):
		if not (self.y and self.yNonLinear): return
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
		return [self.y, self.yLinear, self.yNonLinear, self.deltaLinear, self.deltaNonLinear]


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

	def __init__(self, factorsCount):
		self.factorsCount = factorsCount
		self.main = []
		self.combs = []
		self.colsCount = self.rowsCount = 2 ** factorsCount

		prod = list(product([-1, 1], repeat=factorsCount))
		for i in range(factorsCount + 1):
			if i == 0:
				val = [1 for j in range(2 ** factorsCount)]
			else:
				val = [p[i-1] for p in prod]
			self.main.append(val)

		idx = [i + 1 for i in range(factorsCount)]
		combs = list(combinations(idx, 2)) + \
			list(combinations(idx, 3)) + list(combinations(idx, 4))
		for comb in combs:
			val = []
			for elem in comb:
				val = multTwoLists(val, self.main[elem])
			self.combs.append(Combination(comb, val))

		self.labels = ['x' + str(i) for i in range(self.factorsCount + 1)]
		for comb in combs:
			label = ''
			for elem in comb:
				label += 'x' + str(elem)
			self.labels.append(label)

		self._full = []
		for col in self.main:
			self._full.append(col)
		for comb in self.combs:
			self._full.append(comb.value)

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
	return [elem1 * elem2 for elem1, elem2 in zip(list1, list2)]

def subtractTwoLists(list1: list[int], list2: list[int]):
	if len(list1) == 0:
		return list2
	if len(list2) == 0:
		return list1
	return [elem1 - elem2 for elem1, elem2 in zip(list1, list2)]

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
	naturFactorMatrix: list[list[int]]

	intervals: list[float]

	xFields: list[Entry]
	yFields: list[Entry]

	genTimeGeneratorType: any
	procTimeGeneratorType: any

	def __init__(self, genTimeGeneratorType: any, procTimeGeneratorType: any):
		self.factorsCount = countFactorsByGeneratorType(genTimeGeneratorType) + countFactorsByGeneratorType(procTimeGeneratorType)
		self.xColsCount = 2 ** self.factorsCount
		self.yColsCount = 5
		self.rowsCount = 2 ** self.factorsCount + 1

		self.genTimeGeneratorType = genTimeGeneratorType
		self.procTimeGeneratorType = procTimeGeneratorType

		self.xMatrix = XMatrix(self.factorsCount)
		self.normFactorMatrix = self.xMatrix.factor()
		self.yMatrix = YMatrix(self.factorsCount)

	def grid(self, window, column=0, row=0, columnspan=1, rowspan=1, padx=0, pady=0):
		self.window = window
		self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = column, row, columnspan, rowspan, padx, pady

		self.frame = Frame(window)
		self.frame.grid(row=row, column=column, rowspan=rowspan,
						columnspan=columnspan, sticky=NS, padx=10, pady=10)
		self.frame.grid_rowconfigure(0, weight=1)
		self.frame.grid_columnconfigure(0, weight=1)

		self.__createXLabels()
		self.__createXFields()

		self.__createYLabels()
		self.__createYFields()

	def __createCell(self, row: int, column: int, width: int, color: str, value=None):
		cell = Entry(self.frame, highlightthickness=1,
					 relief=FLAT, justify=CENTER, width=width)
		if value != None:
			cell.insert(0, str(value))
		cell.config(state='readonly', readonlybackground=color)
		cell.grid(row=row, column=column, sticky=EW, padx=0, pady=0)
		return cell

	def __createXLabels(self):
		for j, label in enumerate(self.xMatrix.labels):
			# для ширины колонок по длине содержимого (x-колонки):
			colWidth = len(label)
			# для фиксированной ширины колонок (x-колонки):
			# colWidth = XMATRIX_CELL_WIDTH
			self.__createCell(row=0, column=j+1, width=colWidth,
							  color=LABEL_CELL_COLOR, value=label)

		for i in range(self.xMatrix.rowsCount + 1):
			value = None
			if i != 0:
				value = i
			self.__createCell(
				row=i, column=0, width=NUMBER_CELL_WIDTH, color=LABEL_CELL_COLOR, value=value)

	def __createXFields(self):
		self.xFields = []
		for j in range(self.xMatrix.colsCount):
			col = []
			colWidth = len(self.xMatrix.labels[j])
			for i in range(self.xMatrix.rowsCount):
				cell = self.__createCell(
					row=i+1, column=j+1, width=colWidth, color=COMMON_CELL_COLOR, value=self.xMatrix.full()[j][i])
				col.append(cell)
		self.xFields.append(col)

	def __destroyYFields(self):
		for i in range(len(self.yFields)):
			for j in range(len(self.yFields[0])):
				self.yFields[i][j].destroy()

	def __createYLabels(self):
		offset = self.xMatrix.colsCount
		for j, label in enumerate(self.yMatrix.labels):
			# для ширины колонок по длине содержимого (y-колонки):
			# colWidth = len(label)
			# для фиксированной ширины колонок (y-колонки):
			colWidth = YMATRIX_CELL_WIDTH
			self.__createCell(row=0, column=j+1+offset,
							  width=colWidth, color=LABEL_CELL_COLOR, value=label)

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
					row=i+1, column=j+1+offset, width=colWidth, color=COMMON_CELL_COLOR, value=value)
				col.append(cell)
			self.yFields.append(col)

	def __makeNormFactorMatrixNatural(self, normFactorMatrix):
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

	def setFactorsIntervals(self, intervals: list[Interval]):
		self.intervals = intervals
		self.naturFactorMatrix = self.__makeNormFactorMatrixNatural(
			self.normFactorMatrix)

	def calculateNormNonLinearCoefs(self):
		coefs = []
		for j in range(self.xMatrix.colsCount):
			coef = 0
			for i in range(self.xMatrix.rowsCount):
				coef += self.xMatrix.full()[j][i] * self.yMatrix.y[i]
			coefs.append(coef / self.xMatrix.rowsCount)
		return coefs

	def calculateInterval(self):
		self.yMatrix.calculateY(factorMatrix=self.naturFactorMatrix, time=100, genTimeGeneratorType=self.genTimeGeneratorType, procTimeGeneratorType=self.procTimeGeneratorType)
		self.__destroyYFields()

		normNonLinearCoefs = self.calculateNormNonLinearCoefs()
		normLinearCoefs = normNonLinearCoefs[:(self.factorsCount + 1)]
		
		self.yMatrix.calculateYLinear(normLinearCoefs, self.xMatrix.main)
		self.yMatrix.calculateYNonLinear(normNonLinearCoefs, self.xMatrix.full())
		self.yMatrix.calculateDeltaLinear()
		self.yMatrix.calculateDeltaNonLinear()
		self.__createYFields()


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
		self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = column, row, columnspan, rowspan, padx, pady

		self.frame = self.__createFrame(window, row=row, column=column, rowspan=rowspan, columnspan=columnspan)

		self.genIntensityField = self.genRangeField = self.procIntensityField = self.procRangeField = None

		timeGeneratorsTypes = [self.genTimeGeneratorType, self.procGeneratorType]
		for i, type in enumerate(timeGeneratorsTypes):
			if i == 0: frameText = "Поступление заявок:"
			elif i == 1: frameText = "Обработка заявок:"
			frame = self.__createFrame(self.frame, row=1, column=i+1, text=frameText)

			label = self.__getTimeGeneratorLabel(type)

			lblDistribution = Label(frame, text=label, font=('', 13, "italic"))
			lblDistribution.grid(row=0, column=i, columnspan=3, sticky=EW)

			lblIntensity = Label(frame, text="Интенсивность:")
			lblIntensity.grid(row=2, column=i, sticky=W)

			lblIntensityMin = Label(frame, text="min:")
			lblIntensityMin.grid(row=1, column=i+1, sticky=EW)

			lblIntensityMax = Label(frame, text="max:")
			lblIntensityMax.grid(row=1, column=i+2, sticky=EW)

			txtIntensityMin = self.__createEntry(frame, row=2, column=i+1, sticky=E)
			txtIntensityMax = self.__createEntry(frame, row=2, column=i+2, sticky=W)
			if i == 0: self.genIntensityField = (txtIntensityMin, txtIntensityMax)
			elif i == 1: self.procIntensityField = (txtIntensityMin, txtIntensityMax)

			if (countFactorsByGeneratorType(type) == 2):
				lblRange = Label(frame, text="Разброс интенсивности:")
				lblRange.grid(row=3, column=i, sticky=W)

				txtRangeMin = self.__createEntry(frame, row=3, column=i+1, sticky=E)
				txtRangeMax = self.__createEntry(frame, row=3, column=i+2, sticky=W)
				if i == 0: self.genRangeField = (txtRangeMin, txtRangeMax)
				elif i == 1: self.procRangeField = (txtRangeMin, txtRangeMax)

			print(self.genIntensityField, self.procIntensityField, self.genRangeField, self.procRangeField)

	def __createEntry(self, window, row=0, column=0, sticky=NSEW, value=None):
		entry = Entry(window, width=10)
		if value != None: entry.insert(0, str(value))
		entry.grid(row=row, column=column, sticky=sticky)
		return entry
			
	def __createFrame(self, window, row=0, column=0, columnspan=1, rowspan=1, text=None):
		if text != None: frame = LabelFrame(window, text=text)
		else: frame = Frame(window)

		frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky=NS, padx=10, pady=10)
		frame.grid_rowconfigure(0, weight=1)
		frame.grid_columnconfigure(0, weight=1)

		return frame
		
	def __getTimeGeneratorLabel(self, timeGeneratorType: any):
		if timeGeneratorType is NormalTimeGenerator: return 'Нормальный закон'
		elif timeGeneratorType is ExponentialTimeGenerator: return 'Экспоненциальный закон'
		elif timeGeneratorType is UniformTimeGenerator: return 'Равномерный закон'
		elif timeGeneratorType is WeibullTimeGenerator: return 'Закон Вейбулла'
		elif timeGeneratorType is RayleighTimeGenerator: return 'Закон Рэлея'

	def factors(self):
		factors = []
		for field in [self.genIntensityField, self.procIntensityField, self.genRangeField, self.procRangeField]:
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
	pass


def calcInterval(planningMatrix):
	planningMatrix.calculateInterval()


def calcPoint():
	pass


window = Tk()

window.title('Лабораторная работа №2')
window.resizable(False, False)

intervalDataBlock1 = IntervalDataBlock(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE)
intervalDataBlock1.grid(window=window, row=1, column=1, rowspan=1, columnspan=2)
intervalDataBlock1.setGenFactors(Interval(1, 20))
intervalDataBlock1.setProcFactors(Interval(50, 100))
				 
mtrx2 = PlanningMatrix(GEN_TIME_GENERATOR_TYPE, PROC_TIME_GENERATOR_TYPE)
mtrx2.grid(window=window, row=1, column=0)
mtrx2.setFactorsIntervals([Interval(1, 20), Interval(50, 100)])
calcInterval(mtrx2)

mtrx3 = PlanningMatrix(RayleighTimeGenerator, UniformTimeGenerator)
mtrx3.grid(window=window, row=2, column=0)
mtrx3.setFactorsIntervals([Interval(1, 20), Interval(50, 100), Interval(5, 10)])
calcInterval(mtrx3)

mtrx4 = PlanningMatrix(UniformTimeGenerator, UniformTimeGenerator)
mtrx4.grid(window=window, row=3, column=0)
mtrx4.setFactorsIntervals([Interval(1, 20), Interval(50, 100), Interval(5, 10), Interval(5, 10)])
calcInterval(mtrx4)

window.mainloop()
