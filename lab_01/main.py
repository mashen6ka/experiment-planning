from tkinter import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algs import Model, UniformTimeGenerator, RayleighTimeGenerator, RequestGenerator, RequestProcessor

class Graphics:
	def __init__(self, window, column=0, row=0, columnspan=1, rowspan=1, padx=0, pady=0):
		self.window = window
		self.column, self.row, self.columnspan, self.rowspan, self.padx, self.pady = column, row, columnspan, rowspan, padx, pady
  
		self._canvas = None
		self._setAxis()
  
		self._xDefault = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.060000000000000005, 0.06999999999999999, 0.08, 0.09, 0.09999999999999999, 0.11, 0.12, 0.13, 0.14, 0.15000000000000002, 0.16, 0.17, 0.18000000000000002, 0.19, 0.2, 0.21000000000000002, 0.22, 0.23, 0.24000000000000002, 0.25, 0.26, 0.27, 0.28, 0.29000000000000004, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35000000000000003, 0.36000000000000004, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42000000000000004, 0.43, 0.44, 0.45, 0.46, 0.47000000000000003, 0.48000000000000004, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.5800000000000001, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.6900000000000001, 0.7000000000000001, 0.7100000000000001, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.8400000000000001, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.9600000000000001, 0.97, 0.98, 0.99, 1.0]
		self._yDefault = [0, 0.004767463154206959, 0.0051873173316375875, 0.0050619385718597525, 0.005944256686840961, 0.005424362516367722, 0.006445453406048204, 0.005681842581210706, 0.006449543957590915, 0.007463744772271239, 0.007900252623610693, 0.007650657911051066, 0.008783854511338061, 0.009839459046846822, 0.01044343221364107, 0.01138241097774962, 0.011907551735188555, 0.014165080009609027, 0.014012847650957614, 0.013756042992168522, 0.014644906363534717, 0.015904755447526415, 0.018847072428461505, 0.0193767945322313, 0.022937353736932776, 0.024731295096296714, 0.022741538851499564, 0.027243213870564303, 0.027191736074193812, 0.02874111094544133, 0.029045257551792037, 0.032837984677279455, 0.03492827716163713, 0.03787969289063443, 0.040386815756079254, 0.041764236491887353, 0.04604923059179652, 0.045269981200582905, 0.05177551496724983, 0.050255284295882864, 0.05769272838476027, 0.05875946285395482, 0.05969858893120617, 0.06407338292625488, 0.06811400409641574, 0.06973485474647136, 0.07585699636860688, 0.07823638213940556, 0.08114958366243702, 0.08885293053004625, 0.09524163092176956, 0.0971071236252255, 0.10471813772805084, 0.11343503138039847, 0.11228179307414957, 0.1198896550535707, 0.12265210381854005, 0.13348239867771125, 0.1414376083637546, 0.14572809562423866, 0.15378724767444873, 0.16878280901647177, 0.1690878057145736, 0.17793566632331903, 0.1996876770495842, 0.1950903227426078, 0.20861717505996955, 0.222335099494478, 0.2294416995444396, 0.2426290691003322, 0.26418006632768665, 0.276746107305015, 0.28908179127947325, 0.31265076301128336, 0.338790755424913, 0.347843731612113, 0.37094173881270065, 0.3912425970993242, 0.4151054937302348, 0.43329045501691704, 0.4988041449759798, 0.5216903856987364, 0.5656364053387659, 0.6079385941459858, 0.6454163700549826, 0.6517333119959204, 0.8028208460329078, 0.8146093906791944, 0.8981122122608993, 0.9972385658457533, 1.1779227745805703, 1.2698415885794292, 1.3750052658625886, 1.5120035257819515, 1.8673187644220643, 2.288709029116059, 2.8122968260931493, 3.4439701576481747, 4.522173221399921, 5.544318021066085, 8.01943042662097]
  
	def _setAxis(self):
		self._canvas = None
		self._fig, self._axs = plt.subplots(1, figsize=(8, 4), dpi=50)

		self._fig.suptitle('График зависимости среднего времени ожидания от загрузки системы')

		self._axs.set_xlabel('Загрузка системы')
		self._axs.set_ylabel('Время ожидания')
  
		self._axs.set_xlim(left=0)
		self._axs.set_ylim(bottom=0)

		self._axs.grid(True)

	def plot(self, x=[], y=[]):
		self._setAxis()

		if (len(x) == 0 or len(y) == 0):
			x = self._xDefault
			y = self._yDefault

		yMax = 1 if len(y) == 0 else max(y)
		self._axs.set_ylim(top=yMax)

		self._axs.plot(x, y, color='blue')
		if self._canvas:
			self._canvas.get_tk_widget().destroy()
   

		self._canvas = FigureCanvasTkAgg(self._fig, master=self.window)
		self._canvas.draw()
		self._canvas.get_tk_widget().grid(column=self.column, row=self.row, columnspan=self.columnspan, rowspan=self.rowspan, padx=self.padx, pady=self.pady)

def model():
	genIntensity = float(txtGenIntensity.get())
	procIntensity = float(txtProcIntensity.get())
	procRange = float(txtProcRange.get())
 
	time = int(txtTime.get())
  
	processor = RequestProcessor(UniformTimeGenerator(procIntensity, procRange))
	generator = RequestGenerator(RayleighTimeGenerator(genIntensity), [processor])
	model = Model([generator], [processor])
	result = model.simulate(time)
 
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
  runsNumber = 50
  
  for systemLoad in systemLoadList:
    totalRunsWaitingTime = 0
    
    genIntensity = genBaseIntensity * systemLoad
    procIntensity = procBaseIntensity
    procRange = procBaseRange
    for i in range(runsNumber):
      processor = RequestProcessor(UniformTimeGenerator(procIntensity, procRange))
      generator = RequestGenerator(RayleighTimeGenerator(genIntensity), [processor])
      model = Model([generator], [processor])
      result = model.simulate(time)
      totalRunsWaitingTime += result.processors[0].avgWaitingTime
      
    
    avgWaitingTimeList.append(totalRunsWaitingTime / runsNumber)
  
  print(systemLoadList, '\n')
  print(avgWaitingTimeList)
  
  systemLoadList.insert(0, 0)
  avgWaitingTimeList.insert(0, 0)
  graphics.plot(systemLoadList, avgWaitingTimeList)

window = Tk()

window.title('Лабораторная работа №1')
window.resizable(False, False)

frameMain = LabelFrame(window, text="Исходные данные:", font=('', 13, "bold"))
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

lblGenDistribution = Label(frameGen, text="Закон Рэлея", font=('', 13, "italic"))
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

lblProcDistribution = Label(frameProc, text="Равномерный закон", font=('', 13, "italic"))
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

btnModel = Button(frameMain, text="Построить график", command=lambda:plot(graphics))
btnModel.grid(row=4, column=1, sticky=NSEW, padx=10, pady=10)

frameRes = LabelFrame(window, text="Результаты:", font=('', 13, "bold"))
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
