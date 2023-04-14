import numpy.random as nr
from math import pi
import enum
from queue import PriorityQueue

REQUEST_COUNT = 10000
DELTA_T = 0.01

class UniformTimeGenerator:
  def __init__(self, intensity, range):
    self._a = 1 / intensity - range
    self._b = 1 / intensity + range

  def randomTime(self):
    return nr.uniform(self._a, self._b)
  
class RayleighTimeGenerator:
  def __init__(self, intensity):
    self._sigma = (1 / intensity) * (pi / 2) ** (-1 / 2)

  def randomTime(self):
    return nr.rayleigh(self._sigma)
  
class NormalTimeGenerator:
  def __init__(self, intensity, range):
    self._i = 1 / intensity
    self._range = range

  def randomTime(self):
    return nr.normal(self._i, self._range)
    
class ExponentialTimeGenerator:
  def __init__(self, intensity):
    self._scale = 1 / intensity

  def randomTime(self):
    return nr.exponential(self._scale)
  
class WeibullTimeGenerator:
  def __init__(self, sigma):
    self._sigma = sigma
  
  def randomTime(self):
    return nr.weibull(self._sigma)

class Request:
  def __init__(self):
    self.timeIn = None
    self.timeOut = None

class RequestGenerator:
  def __init__(self, timeGenerator, receivers = []):
    self._timeGenerator = timeGenerator
    self._receivers = receivers
    self._request = None
    self._busy = False
    
    self.totalRequests = 0
    self.totalGenerationTime = 0

  def startGeneration(self, currTime):
    if self._busy: return None
    self._busy = True
    duration = self.generateDuration()
    next = currTime + duration
    self.totalGenerationTime += duration
    
    self._request = Request()
    self._request.timeIn = next
    return next

  def finishGeneration(self):
    if not self._busy: return None
    self._busy = False
    self.totalRequests += 1
    
    minQueueSize = self._receivers[0].queueSize
    minReceiverId = 0
    for index, receiver in enumerate(self._receivers):
      if receiver.queueSize < minQueueSize:
        minQueueSize = receiver.queueSize
        minReceiverId = index

    self._receivers[minReceiverId].pushRequest(self._request)
    return self._receivers[minReceiverId]

  def generateDuration(self):
    return self._timeGenerator.randomTime()

class RequestProcessor:
  def __init__(self, timeGenerator):
    self._timeGenerator = timeGenerator
    self._queue = []
    self._waitingTime = 0
    self._busy = False
    
    self.totalRequests = 0
    self.totalProcessingTime = 0
    self.totalWaitingTime = 0
  
  @property
  def queueSize(self):
    return len(self._queue)

  def pushRequest(self, request):
    self._queue.append(request)
    
  def startProcessing(self, currTime):
    if self._busy or len(self._queue) == 0: return None
    self._busy = True
    request = self._queue.pop(0)
    duration = self.generateDuration()
    next = currTime + duration
    
    request.timeOut = next
    self.totalWaitingTime += currTime - request.timeIn
    
    self.totalProcessingTime += duration
    return next
    
  def finishProcessing(self):
    if not self._busy: return None
    self._busy = False
    self.totalRequests += 1

  def generateDuration(self):
    return self._timeGenerator.randomTime()
      
class ModellingResult:
  class GeneratorResult:
    def __init__(self, generator, index):
      self.index = index
      self.totalRequests = generator.totalRequests
      self.totalGenerationTime = generator.totalGenerationTime
      
      self.avgGenerationTime = self.totalGenerationTime / self.totalRequests
  
  class ProcessorResult:
    def __init__(self, processor, index):
      self.index = index
      self.totalRequests = processor.totalRequests
      self.totalProcessingTime = processor.totalProcessingTime
      self.totalWaitingTime = processor.totalWaitingTime
      
      self.avgProcessingTime = self.totalProcessingTime / self.totalRequests
      self.avgWaitingTime= self.totalWaitingTime / self.totalRequests

  def __init__(self, generators, processors):
    self.generators = []
    self.processors = []
    
    for index, generator in enumerate(generators, start=1):
      self.generators.append(self.GeneratorResult(generator, index))
    
    for index, processor in enumerate(processors, start=1):
      self.processors.append(self.ProcessorResult(processor, index))

class Model:
  def __init__(self, generators, processors):
    self.generators = generators
    self.processors = processors
    self._eventList = PriorityQueue()
 
  class EventType(enum.Enum):
    simulationFinished = 0
    genFinished = 1
    procFinished = 2
    
  class Event:
    def __init__(self, time, type, block=None):
      self.time = time
      self.type = type
      self.block = block
      
  def addEvent(self, eventNew):
    self._eventList.put((eventNew.time, eventNew))

  def simulateEventBased(self, maxTime):
    self._eventList = PriorityQueue()
    self.addEvent(self.Event(maxTime, self.EventType.simulationFinished))

    for generator in self.generators:
      next = generator.startGeneration(0)
      self.addEvent(self.Event(next, self.EventType.genFinished, generator))
    
    while not self._eventList.empty():
      _, event = self._eventList.get()
      if event.type == self.EventType.simulationFinished: break
      
      if event.type == self.EventType.genFinished:
        generator = event.block
        processor = generator.finishGeneration()
        next = generator.startGeneration(event.time)
        self.addEvent(self.Event(next, self.EventType.genFinished, generator))
        
        next = processor.startProcessing(event.time)
        if (next): self.addEvent(self.Event(next, self.EventType.procFinished, processor))

      if event.type == self.EventType.procFinished:
        processor = event.block
        processor.finishProcessing()
        next = processor.startProcessing(event.time)
        if (next): self.addEvent(self.Event(next, self.EventType.procFinished, processor))
      
    for processor in self.processors:
      processor.totalProcessingTime = min(processor.totalProcessingTime, maxTime)

    for generator in self.generators:
      generator.totalGenerationTime = min(generator.totalGenerationTime, maxTime)
    
    return ModellingResult(self.generators, self.processors)