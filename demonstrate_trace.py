#!/usr/bin/env python
"""
Demonstrate CDF produced by JobGenerator.
Calls generate_trace,
unpacks generate samples
plots model_size, itterations, arrival and duration CDFs
"""
import matplotlib.pyplot as plt
import numpy as np
from core.jobs.job_generator import JobGenerator
__author__ = "Dominic Lindsay (Babbleshack)"
__email__ = "dcrl94@gmail.com"

debug = True

number = 200
jg = JobGenerator()

trace = jg.generate_trace(number)
assert(trace)
assert(any(trace["model"]))
assert(any(trace["itter"]))
assert(any(trace["arrival"]))
assert(any(trace["duration"]))
assert(any(trace["num_gpu"]))
assert(any(trace["interval"]))

model_samples = trace["model"]
itter_samples = trace["itter"]
arrival_samples = trace["arrival"]
duration_samples = trace["duration"]
num_gpu = trace["num_gpu"]
interval = trace["interval"]
#model_samples = jg.generate_samples(number, jg.get_model_samples(), jg.get_model_distribution())
#itter_samples = jg.generate_samples(number, jg.get_itter_samples(), jg.get_itter_distribution())
#arrival_samples = jg.generate_samples(number, jg.get_arrival_samples(), jg.get_arrival_distribution())
#duration_samples = jg.generate_samples(number, jg.get_duration_samples(), jg.get_duration_distribution())

#Samples must be sorted before plotting
model_samples_s = np.sort(model_samples)
itter_samples_s = np.sort(itter_samples)
arrival_samples_s = np.sort(arrival_samples)
duration_samples_s = np.sort(duration_samples)

if debug:
    print(trace)

#Plot samples
plt.figure()
plt.suptitle("CDF ({} generated samples)".format(number))

model_plot = plt.subplot(221)
itter_plot = plt.subplot(222)
arrival_plot = plt.subplot(223)
duration_plot = plt.subplot(224)

model_plot.plot(model_samples_s, jg.cdf(model_samples))
model_plot.set_title("Model Sizes")
itter_plot.plot(itter_samples_s, jg.cdf(itter_samples))
itter_plot.set_title("Itterations")
arrival_plot.plot(arrival_samples_s, jg.cdf(arrival_samples))
arrival_plot.set_title("Job Arrival")
duration_plot.plot(duration_samples_s, jg.cdf(duration_samples))
duration_plot.set_title("Duration (Hours)")

plt.show()
