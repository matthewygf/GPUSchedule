import numpy as np
from random import choices

class JobGenarator(object):
    """
    A Job Generator
    Generates jobs from known distributions.
    """
    def __init__(self):
        #Default Samples
        self.MODEL_SAMPLE_DEF = []
        self.ITTER_SAMPLE_DEF = [1,1,1,1,1,1, 109, 126, 133, 138, 141, 143, 144, 147,
                                 157, 168, 175, 192, 193, 198, 235, 237, 242, 253,
                                 258, 272, 272, 274, 288, 326, 326, 362, 386, 391,
                                 410, 438, 447, 468, 473, 513, 513, 521, 521, 525,
                                 581, 606, 607, 775, 775, 789, 822, 864, 864, 892,
                                 903, 949, 1011, 1085, 1360, 1501, 2178, 2239, 2275,
                                 3304, 3469, 4861]

        self.DURATION_SAMPLE_DEF = [121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121,
                                    121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122,
                                    122, 123, 123, 123, 124, 125, 125, 126, 126, 126, 126, 127,
                                    128, 130, 131, 133, 135, 138, 141, 143, 147, 152, 155, 158,
                                    164, 171, 180, 189, 196, 209, 230, 263, 305, 368, 536,
                                    1800]
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
        cdf = 1. * np.arange(len(cdf)) / (len(cdf) - 1)
        return cdf

    def set_model_distribution(self, samples):
        cdf = self.cdf(samples)
        self.model_samples = samples
        self.model_distribution = cdf

    def set_duration_distribution(self, samples):
        cdf = self.cdf(samples)
        self.duration_samples = samples
        self.duration_distribution = cdf

    def set_itter_distribution(self, samples):
        cdf = self.cdf(samples)
        self.itter_samples = samples
        self.itter_distribution = cdf

    def set_arrival_distribution(self, samples):
        cdf = self.cdf(samples)
        self.arrival_samples = samples
        self.arrival_distribution = cdf

    def generate_model_samples(self, number):
        """Generate models based from CDF
        """
        if not self.model_samples or not self.model_distribution:
            raise Exception("Error model distribution/samples not set")
        elif number < len(self.model_samples):
            print("WARN: generating fewer samples then distribution")
        samples = choices(self.model_samples,
                          cum_weights=self.model_distribution, k=number)
        return samples

    def generate_duration_samples(self, number):
        """Generate durations based from CDF
        """
        if not self.duration_samples or not self.duration_distribution:
            raise Exception("Error duration distribution/samples not set")
        elif number < len(self.duration_samples):
            print("WARN: generating fewer samples then distribution")
        samples = choices(self.duration_samples,
                          cum_weights=self.duration_distribution, k=number)
        return samples

    def generate_itter_samples(self, number):
        """Generate itters based from CDF
        """
        if not self.itter_samples or not self.itter_distribution:
            raise Exception("Error itter distribution/samples not set")
        elif number < len(self.itter_samples):
            print("WARN: generating fewer samples then distribution")
        samples = choices(self.itter_samples,
                          cum_weights=self.itter_distribution, k=number)
        return samples

    def generate_arrival_samples(self, number):
        """Generate arrivals based from CDF
        """
        if not self.arrival_samples or not self.arrival_distribution:
            raise Exception("Error arrival distribution/samples not set")
        elif number < len(self.arrival_samples):
            print("WARN: generating fewer samples then distribution")
        samples = choices(self.arrival_samples,
                          cum_weights=self.arrival_distribution, k=number)
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
        trace["model"] = self.generate_model_samples(number)
        trace["duration"] = self.generate_duration_samples(number)
        trace["itter"] = self.generate_itter_samples(number)
        trace["arrival"] = self.generate_arrival_samples(number)
        return trace
