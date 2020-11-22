"""
Job Generator, generates traces from sample population and distributions
"""
import logging
import numpy as np
import random
import pandas as pd
import sys
import os

__author__ = "Dominic Lindsay (Babbleshack)"
__email__ = "dcrl94@gmail.com"

class JobGenerator(object):
    """
    A Job Generator
    Generates jobs from known distributions.
    """
    def __init__(self):
        #Default Samples
        self.MODEL_SAMPLE_DEF = [1.04, 1.27, 15.36, 108.48, 125.16, 125.16,
                                 237.61, 330.52, 364.2, 462.77, 544.24, 636.75,
                                 664.1, 700., 798.28, 862.95]
        self.ITTER_SAMPLE_DEF = [1,1,1,1,1,1, 109, 126, 133, 138, 141, 143,
                                 144, 147, 157, 168, 175, 192, 193, 198, 235,
                                 237, 242, 253, 258, 272, 272, 274, 288, 326,
                                 326, 362, 386, 391, 410, 438, 447, 468, 473,
                                 513, 513, 521, 521, 525, 581, 606, 607, 775,
                                 775, 789, 822, 864, 864, 892, 903, 949, 1011,
                                 1085, 1360, 1501, 2178, 2239, 2275, 3304,
                                 3469, 4861]
        self.DURATION_SAMPLE_DEF = [121, 121, 121, 121, 121, 121, 121, 121,
                                    121, 121, 121, 121, 121, 121, 121, 121,
                                    121, 121, 121, 121, 121, 122, 122, 122,
                                    122, 123, 123, 123, 124, 125, 125, 126,
                                    126, 126, 126, 127, 128, 130, 131, 133,
                                    135, 138, 141, 143, 147, 152, 155, 158,
                                    164, 171, 180, 189, 196, 209, 230, 263,
                                    305, 368, 536, 1800]
        self.ARRIVAL_SAMPLE_DEF = [10, 100, 500,1000, 3000, 5000, 10000]
        #Default Distributions
        self.MODEL_DIST_DEF = self.cdf(self.MODEL_SAMPLE_DEF)
        self.ITTER_DIST_DEF = self.cdf(self.ITTER_SAMPLE_DEF)
        self.ARRIVAL_DIST_DEF = self.cdf(self.ARRIVAL_SAMPLE_DEF)
        self.DURATION_DIST_DEF = self.cdf(self.DURATION_SAMPLE_DEF)
        #Initi member vars
        self.model_distribution = self.MODEL_SAMPLE_DEF
        self.model_samples = self.MODEL_SAMPLE_DEF
        self.duration_distribution = self.DURATION_DIST_DEF
        self.duration_samples = self.DURATION_SAMPLE_DEF
        self.itter_distribution = self.ITTER_DIST_DEF
        self.itter_samples = self.ITTER_SAMPLE_DEF
        self.arrival_distribution = self.ARRIVAL_DIST_DEF
        self.arrival_samples = self.ARRIVAL_SAMPLE_DEF

    def cdf(self, samples):
        """Calculate Cumalative Distribution Function of Samples
        """
        samples_s = np.sort(samples)
        cdf = 1. * np.arange(len(samples)) / (len(samples) - 1)
        return cdf

    def get_model_distribution(self):
        return self.model_distribution

    def get_duration_distribution(self):
        return self.duration_distribution

    def get_itter_distribution(self):
        return self.itter_distribution

    def get_arrival_distribution(self):
        return self.arrival_distribution

    def get_model_samples(self):
        return self.model_samples

    def get_duration_samples(self):
        return self.duration_samples

    def get_itter_samples(self):
        return self.itter_samples

    def get_arrival_samples(self):
        return self.arrival_samples

    def set_model_distribution(self, samples):
        """Set model distribution.
        Calculate cumaltive distribution over sample population
        samples: list, a list of samples
        also set population to sample
        """
        cdf = self.cdf(samples)
        self.model_samples = samples
        self.model_distribution = cdf

    def set_duration_distribution(self, samples):
        """Set duration distribution.
        Calculate cumaltive distribution over sample population
        samples: list, a list of samples
        also set population to sample
        """
        cdf = self.cdf(samples)
        self.duration_samples = samples
        self.duration_distribution = cdf

    def set_itter_distribution(self, samples):
        """Set itter distribution.
        Calculate cumaltive distribution over sample population
        samples: list, a list of samples
        also set population to sample
        """
        cdf = self.cdf(samples)
        self.itter_samples = samples
        self.itter_distribution = cdf

    def set_arrival_distribution(self, samples):
        """Set arrival distribution.
        Calculate cumaltive distribution over sample population
        also set population to sample
        samples: list, a list of samples
        """
        cdf = self.cdf(samples)
        self.arrival_samples = samples
        self.arrival_distribution = cdf

    def generate_samples(self, number, population, distribution):
        """Generate samples based on population and distribution
        """
        if not any(population) or not any(distribution):
            raise Exception("Error population and samples cannot be null")
        elif number < len(population):
            print("WARN: generating fewer samples then population")
        samples = random.choices(population,
                          cum_weights=distribution, k=number)
        return samples

    def generate_trace(self, number):
        """Generate Job Trace
        Returns dictionary
        {
            "model": samples - list,
            "duration": samples - list,
            "itter": samples - list,
            "arrival": samples - list
        }"""
        trace = {}
        trace["model"] = self.generate_samples(number, self.model_samples, self.model_distribution)
        trace["duration"] = self.generate_samples(number, self.duration_samples, self.duration_distribution)
        trace["itter"] = self.generate_samples(number, self.itter_samples, self.itter_distribution)
        trace["arrival"] = self.generate_samples(number, self.arrival_samples, self.arrival_distribution)
        #TODO: update to normal distribition
        trace["num_gpu"] = []
        trace["interval"] = []
        trace["job_id"] = []
        for i in range(number):
            trace["num_gpu"].append(random.randrange(1, 128))
        for i in range(number):
            trace["interval"].append(random.randrange(20, 44))
        for i in range(number):
            trace["job_id"].append(i)
        return trace


class JobTraceReader(object):
    """
    A reader that takes into a trace file and iterate through the rows
    """
    def __init__(self, file_path) -> None:
        if not os.path.exists(file_path):
            logging.error(f"file: {file_path} not exist")
            sys.exit(1)

        try:
            self.trace_df = pd.read_csv(file_path)
            self.delta_time_offset = 0
        except :
            logging.error(f"unable to read the file, assumed it was csv. but got {file_path}")
            sys.exit(1)
    
    def prepare_jobs(self):
        logging.info("original: %d" % len(self.trace_df))
        self.trace_df = self.trace_df[self.trace_df["type"] == "noninteractive"]
        logging.info("only batch: %d" % len(self.trace_df))
        self.trace_df.sort_values(by="normalized_time", inplace=True)
        self.trace_df.dropna(inplace=True)
        logging.info("dropped nan: %d" % len(self.trace_df))
        self.trace_df["normalized_time"] -= self.trace_df["normalized_time"].min()
        self.trace_df["normalized_time"] /= 10000
        # self.trace_df.to_csv("normalized_submit.csv",index=False)
        
        logging.info("Finished Prepping...")
        self.trace_df["generated"] = 0
    
    def remaining_jobs(self):
        return len(self.trace_df[self.trace_df["generated"] == 0])

    def generate_jobs(self, delta_time):
        '''output jobs that are not scheduled and already passed the delta time'''
        def _gen(df, delta):
            # not yet generated + passed the time offset
            logging.info("delta-time: %d" % delta)
            cond = (df["generated"] == 0) & (df["normalized_time"] <= delta)
            return df[cond]
        to_be_generated = _gen(self.trace_df, delta_time)
        self.trace_df.loc[to_be_generated.index, "generated"] = 1
        return to_be_generated