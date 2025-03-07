from abc import ABC, abstractmethod

class VehicleTrackerPlugin(ABC):
    """ Abstract base class for vehicle tracking plugins. """

    @abstractmethod
    def track_vehicle(self, step):
        """ Tracks the selected vehicle at each simulation step. """
        pass

    @abstractmethod
    def get_summary(self):
        """ Returns a summary of the tracked vehicle's journey. """
        pass
