import numpy.random as nr
from math import pi

REQUEST_COUNT = 10000
DELTA_T = 0.01

class UniformTimeGenerator:
  def __init__(self, intensity, range):
    self._a = 1 / intensity - range
    self._b = 1 / intensity + range

  def randomTime(self):
    return nr.uniform(self._a, self._b)
  
# class NormalTimeGenerator:  
#   def __init__(self, intensity, range):
#     self._i = 1 / intensity
#     self._range = range
  
#   def randomTime(self):
#     return nr.normal(self._i, self._range) 
  
class RayleighTimeGenerator:
  def __init__(self, intensity):
    self._sigma = (1 / intensity) * (pi / 2) ** (-1 / 2)

  def randomTime(self):
    return nr.rayleigh(self._sigma)

class Request:
  def __init__(self):
    self.timeIn = None
    self.timeOut = None

class RequestGenerator:
  def __init__(self, timeGenerator, receivers = []):
    self._timeGenerator = timeGenerator
    self._receivers = receivers
    self._next = 0
    
    self.totalRequests = 0
    self.totalGenerationTime = 0
  
  @property
  def next(self):
    return self._next

  def generateRequest(self, currTime):
    self.totalRequests += 1
    duration = self.generateDuration()
    self._next = currTime + duration
    self.totalGenerationTime += duration
    
    minQueueSize = self._receivers[0].queueSize
    minReceiverId = 0
    for index, receiver in enumerate(self._receivers):
      if receiver.queueSize < minQueueSize:
        minQueueSize = receiver.queueSize
        minReceiverId = index

    request = Request()
    request.timeIn = self._next
    self._receivers[minReceiverId].pushRequest(request)

  def generateDuration(self):
    return self._timeGenerator.randomTime()

class RequestProcessor:
  def __init__(self, timeGenerator):
    self._timeGenerator = timeGenerator
    self._queue = []
    self._next = 0
    self._waitingTime = 0
    
    self.totalRequests = 0
    self.totalProcessingTime = 0
    self.totalWaitingTime = 0
  
  @property
  def next(self):
    return self._next

  @next.setter
  def next(self, value):
    self._next = value
  
  @property
  def queueSize(self):
    return len(self._queue)

  def pushRequest(self, request):
    self._queue.append(request)

  def popRequest(self, currTime):
    if len(self._queue) > 0 and self._queue[0].timeIn <= currTime:
      request = self._queue.pop(0)
      duration = self.generateDuration()
      self._next = currTime + duration
      
      request.timeOut = self._next
      self.totalWaitingTime += currTime - request.timeIn
      
      self.totalProcessingTime += duration
      self.totalRequests += 1
      
      return request

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

  def simulate(self, maxTime):    
    blocks = [*self.generators, *self.processors]

    currTime = 0
    while currTime < maxTime:
      for block in blocks:
        if block.next <= currTime:
          if isinstance(block, RequestGenerator):
            block.generateRequest(currTime)
          if isinstance(block, RequestProcessor):
            block.popRequest(currTime)
      currTime += DELTA_T
      
    for processor in self.processors:
      processor.totalProcessingTime = min(processor.totalProcessingTime, maxTime)

    for generator in self.generators:
      generator.totalProcessingTime = min(generator.totalGenerationTime, maxTime)
    
    # print('Total generated: ', self.generators[0].totalRequests)
    # print('Total generation time: ', self.generators[0].totalGenerationTime)
    # print('Avg generation time: ', self.generators[0].totalGenerationTime / self.generators[0].totalRequests)
    
    # print('Total processed: ', self.processors[0].totalRequests)
    # print('Total processing time: ', self.processors[0].totalProcessingTime)
    # print('Avg processing time: ', self.processors[0].totalProcessingTime / self.processors[0].totalRequests)
    # print('Avg waiting time: ', self.processors[0].totalWaitingTime / self.processors[0].totalRequests)
    
    return ModellingResult(self.generators, self.processors)

