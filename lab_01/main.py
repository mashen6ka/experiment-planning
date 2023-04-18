from tkinter import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algs import (
    Model,
    UniformTimeGenerator,
    RayleighTimeGenerator,
    RequestGenerator,
    RequestProcessor,
)


class Graphics:
    def __init__(self, window, column=0, row=0, columnspan=1, rowspan=1, padx=0, pady=0):
        self.window = window
        self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = (
            column,
            row,
            columnspan,
            rowspan,
            padx,
            pady,
        )

        self._canvas = None
        self._setAxis()

        self._xDefault = [
            0,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.060000000000000005,
            0.06999999999999999,
            0.08,
            0.09,
            0.09999999999999999,
            0.11,
            0.12,
            0.13,
            0.14,
            0.15000000000000002,
            0.16,
            0.17,
            0.18000000000000002,
            0.19,
            0.2,
            0.21000000000000002,
            0.22,
            0.23,
            0.24000000000000002,
            0.25,
            0.26,
            0.27,
            0.28,
            0.29000000000000004,
            0.3,
            0.31,
            0.32,
            0.33,
            0.34,
            0.35000000000000003,
            0.36000000000000004,
            0.37,
            0.38,
            0.39,
            0.4,
            0.41000000000000003,
            0.42000000000000004,
            0.43,
            0.44,
            0.45,
            0.46,
            0.47000000000000003,
            0.48000000000000004,
            0.49,
            0.5,
            0.51,
            0.52,
            0.53,
            0.54,
            0.55,
            0.56,
            0.5700000000000001,
            0.5800000000000001,
            0.59,
            0.6,
            0.61,
            0.62,
            0.63,
            0.64,
            0.65,
            0.66,
            0.67,
            0.68,
            0.6900000000000001,
            0.7000000000000001,
            0.7100000000000001,
            0.72,
            0.73,
            0.74,
            0.75,
            0.76,
            0.77,
            0.78,
            0.79,
            0.8,
            0.81,
            0.8200000000000001,
            0.8300000000000001,
            0.8400000000000001,
            0.85,
            0.86,
            0.87,
            0.88,
            0.89,
            0.9,
            0.91,
            0.92,
            0.93,
            0.9400000000000001,
            0.9500000000000001,
            0.9600000000000001,
            0.97,
            0.98,
            0.99,
            1.0,
        ]
        self._yDefault = [
            0,
            0.0,
            1.114182341801495e-05,
            0.0005770287614270518,
            0.00047037953194210987,
            0.0009609916058847859,
            0.0008878653336763816,
            0.0008001452633552154,
            0.001160125088034971,
            0.0022376958405691113,
            0.002753804082026125,
            0.003367124110628202,
            0.003645419626646032,
            0.0044144478544747906,
            0.004805615400410283,
            0.005769842420740522,
            0.007116380000999194,
            0.006987404613011253,
            0.008457894331440098,
            0.010258458477237971,
            0.01066293887226873,
            0.011440810587443395,
            0.013593812927518451,
            0.014101797455247748,
            0.01610279649199665,
            0.0178064966658927,
            0.018651415162970163,
            0.01941992272327428,
            0.02303347712084883,
            0.02428650504460805,
            0.026332904349599548,
            0.026640268658213836,
            0.029819500245147186,
            0.031114336744576265,
            0.0345562603136411,
            0.0371361261879136,
            0.03972078921776731,
            0.04065018069285824,
            0.043930286064988874,
            0.04564305324816055,
            0.050154589180993824,
            0.05296438701653489,
            0.055722836788743804,
            0.05828415143368801,
            0.0637052299750786,
            0.0678015548713154,
            0.07047490064986058,
            0.07453218387015491,
            0.08041274230457905,
            0.08217683311562704,
            0.08838802277579358,
            0.09408671355174456,
            0.09670747384631075,
            0.10111276682153157,
            0.10866814809285293,
            0.11408941274024223,
            0.12145133127163434,
            0.12595167285049155,
            0.13300486917151383,
            0.14256061475982926,
            0.1476088080648543,
            0.1575844805191144,
            0.16309850519313002,
            0.17670857227721107,
            0.183012848856146,
            0.19198502716374477,
            0.20500385187928882,
            0.21412854608216145,
            0.22543972938012316,
            0.24072638989209255,
            0.2533873764654497,
            0.27367831067453335,
            0.2873781731236446,
            0.30425051182969776,
            0.32698500695566174,
            0.34156254560967314,
            0.361629464844921,
            0.39945687568366955,
            0.40473225338800867,
            0.4414077986922559,
            0.474696805851344,
            0.5145697741717545,
            0.5490044713117945,
            0.5816722010198285,
            0.6296943784361908,
            0.7023877904379979,
            0.7468022362857019,
            0.8204050325723942,
            0.9364351841032155,
            0.9872468275115684,
            1.1105289042852784,
            1.285184331574748,
            1.4200563404637712,
            1.7245827500139002,
            1.9677439247388264,
            2.4417892802019843,
            2.9315210293040277,
            3.5768271443195077,
            4.662829141830647,
            6.4864580547015445,
            8.691542113622038,
        ]

    def _setAxis(self):
        self._canvas = None
        self._fig, self._axs = plt.subplots(1, figsize=(8, 4), dpi=50)

        self._fig.suptitle("График зависимости среднего времени ожидания от загрузки системы")

        self._axs.set_xlabel("Загрузка системы")
        self._axs.set_ylabel("Время ожидания")

        self._axs.set_xlim(left=0)
        self._axs.set_ylim(bottom=0)

        self._axs.grid(True)

    def plot(self, x=[], y=[]):
        self._setAxis()

        if len(x) == 0 or len(y) == 0:
            x = self._xDefault
            y = self._yDefault

        yMax = 1 if len(y) == 0 else max(y)
        self._axs.set_ylim(top=yMax)

        self._axs.plot(x, y, color="blue")
        if self._canvas:
            self._canvas.get_tk_widget().destroy()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self.window)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(
            column=self.column,
            row=self.row,
            columnspan=self.columnspan,
            rowspan=self.rowspan,
            padx=self.padx,
            pady=self.pady,
        )


def model():
    genIntensity = float(txtGenIntensity.get())
    procIntensity = float(txtProcIntensity.get())
    procRange = float(txtProcRange.get())

    time = int(txtTime.get())

    processor = RequestProcessor(UniformTimeGenerator(procIntensity, procRange))
    generator = RequestGenerator(RayleighTimeGenerator(genIntensity), [processor])
    model = Model([generator], [processor])
    result = model.simulateEventBased(time)

    lambdaReal = 1 / (result.generators[0].totalGenerationTime / result.generators[0].totalRequests)
    muReal = 1 / (result.processors[0].totalProcessingTime / result.processors[0].totalRequests)

    systemLoadReal = lambdaReal / muReal
    systemLoadExp = genIntensity / procIntensity

    txtResExp.config(state="normal")
    txtResExp.delete(0, END)
    txtResExp.insert(0, "{:.2f}".format(systemLoadExp))
    txtResExp.config(state="readonly", readonlybackground="white")

    txtResReal.config(state="normal")
    txtResReal.delete(0, END)
    txtResReal.insert(0, "{:.2f}".format(systemLoadReal))
    txtResReal.config(state="readonly", readonlybackground="white")


def plot(graphics):
    systemLoadList = np.arange(0.01, 1.01, 0.01).tolist()
    avgWaitingTimeList = []

    genBaseIntensity = 1
    procBaseIntensity = 1
    procBaseRange = 0.02

    time = 1000
    runsNumber = 200

    for i, systemLoad in enumerate(systemLoadList):
        totalRunsWaitingTime = 0

        genIntensity = genBaseIntensity * systemLoad
        procIntensity = procBaseIntensity
        procRange = procBaseRange

        for _ in range(runsNumber):
            processor = RequestProcessor(UniformTimeGenerator(procIntensity, procRange))
            generator = RequestGenerator(RayleighTimeGenerator(genIntensity), [processor])
            model = Model([generator], [processor])
            result = model.simulateEventBased(time)
            totalRunsWaitingTime += result.processors[0].avgWaitingTime
        print("Processed {:.0f}%".format((i + 1) / len(systemLoadList) * 100))

        avgWaitingTimeList.append(totalRunsWaitingTime / runsNumber)

    systemLoadList.insert(0, 0)
    avgWaitingTimeList.insert(0, 0)

    print("\n", systemLoadList, "\n\n", avgWaitingTimeList)

    graphics.plot(systemLoadList, avgWaitingTimeList)


window = Tk()

window.title("Лабораторная работа №1")
window.resizable(False, False)

frameMain = LabelFrame(window, text="Исходные данные:", font=("", 13, "bold"))
frameMain.grid(row=0, column=0, sticky=NS, padx=10, pady=10)

lblTime = Label(frameMain, text="Время моделирования:")
lblTime.grid(row=0, column=0, sticky=NSEW, padx=10, pady=10)
txtTime = Entry(frameMain, width=10)
txtTime.insert(0, 1000)
txtTime.grid(row=0, column=1, sticky=NSEW, padx=10, pady=10)

frameGen = LabelFrame(frameMain, text="Поступление заявок:")
frameGen.grid(row=1, column=0, columnspan=2, sticky=NSEW, padx=10, pady=10)
frameGen.grid_rowconfigure(0, weight=1)
frameGen.grid_columnconfigure(0, weight=1)

lblGenDistribution = Label(frameGen, text="Закон Рэлея", font=("", 13, "italic"))
lblGenDistribution.grid(row=0, column=0, columnspan=2, sticky=EW)

lblGenIntensity = Label(frameGen, text="Интенсивность:")
lblGenIntensity.grid(row=1, column=0, sticky=W)
txtGenIntensity = Entry(frameGen, width=10)
txtGenIntensity.insert(0, 1.5)
txtGenIntensity.grid(row=1, column=1, sticky=E)

frameProc = LabelFrame(frameMain, text="Обработка заявок:")
frameProc.grid(row=2, column=0, columnspan=2, sticky=NSEW, padx=10, pady=10)
frameProc.grid_rowconfigure(0, weight=1)
frameProc.grid_columnconfigure(0, weight=1)

lblProcDistribution = Label(frameProc, text="Равномерный закон", font=("", 13, "italic"))
lblProcDistribution.grid(row=0, column=0, columnspan=2, sticky=EW)

lblProcIntensity = Label(frameProc, text="Интенсивность:")
lblProcIntensity.grid(row=1, column=0, sticky=W)
txtProcIntensity = Entry(frameProc, width=10)
txtProcIntensity.insert(0, 0.2)
txtProcIntensity.grid(row=1, column=1, sticky=E)

lblProcRange = Label(frameProc, text="Разброс времени:")
lblProcRange.grid(row=2, column=0, sticky=W)
txtProcRange = Entry(frameProc, width=10)
txtProcRange.insert(0, 3)
txtProcRange.grid(row=2, column=1, sticky=E)

btnModel = Button(frameMain, text="Моделировать", command=model)
btnModel.grid(row=4, column=0, sticky=NSEW, padx=10, pady=10)

btnModel = Button(frameMain, text="Построить график", command=lambda: plot(graphics))
btnModel.grid(row=4, column=1, sticky=NSEW, padx=10, pady=10)

frameRes = LabelFrame(window, text="Результаты:", font=("", 13, "bold"))
frameRes.grid(row=1, column=0, sticky=NSEW, padx=10, pady=10)
frameRes.grid_rowconfigure(0, weight=1)
frameRes.grid_columnconfigure(0, weight=1)

lblResExp = Label(frameRes, text="Расчетная загрузка:")
lblResExp.grid(row=0, column=0, sticky=W, padx=10)
txtResExp = Entry(frameRes, width=10)
txtResExp.config(state="readonly", readonlybackground="white")
txtResExp.grid(row=0, column=1, sticky=E, padx=10)

lblResReal = Label(frameRes, text="Реальная загрузка:")
lblResReal.grid(row=1, column=0, sticky=W, padx=10)
txtResReal = Entry(frameRes, width=10)
txtResReal.config(state="readonly", readonlybackground="white")
txtResReal.grid(row=1, column=1, sticky=E, padx=10)

frameGraph = Frame(window, highlightbackground="gray", highlightthickness=1)
frameGraph.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
graphics = Graphics(frameGraph)
graphics.plot()

window.mainloop()
