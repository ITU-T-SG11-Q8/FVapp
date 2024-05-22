import copy

# Demo libs
import SPIGA.spiga.demo.analyze.extract.processor as pr


class VideoAnalyzer:
    def __init__(self, tracker, processor=pr.EmptyProcessor()):
        self.tracker = tracker
        self.processor = processor
        self.tracked_obj = []

    def process_frame(self, image):
        image = copy.copy(image)
        self.tracked_obj = self.tracker.process_frame(image, self.tracked_obj)
        if len(self.tracked_obj) > 0:
            self.tracked_obj = self.processor.process_frame(image, self.tracked_obj)
        self.tracked_obj = self._add_attributes()
        return self.tracked_obj

    def grm_encode_process_frame(self, image):
        image = copy.copy(image)
        self.tracked_obj, features_tracker = self.tracker.grm_encode_process_frame(image, self.tracked_obj)

        if len(self.tracked_obj) > 0:
            self.tracked_obj, features_spiga = self.processor.grm_encode_process_frame(image, self.tracked_obj)
        if len(self.tracked_obj) == 0:
            features_spiga = None

        self.tracked_obj = self._add_attributes()
        # if features_tracker is None:
        #     print('features_tracker is None')
        # elif features_spiga is None:
        #     print('features_spiga is None')
        # else:
        #     print(f'####### features_tracker:{features_tracker} features_spiga:{features_spiga}')
        return features_tracker, features_spiga

    def grm_decode_process_frame(self, features_tracker, features_spiga):
        self.tracked_obj = self.tracker.grm_decode_process_frame(features_tracker, self.tracked_obj)
        if len(self.tracked_obj) > 0:
            self.tracked_obj = self.processor.grm_decode_process_frame(features_spiga, self.tracked_obj)
        self.tracked_obj = self._add_attributes()
        return self.tracked_obj

    def plot_features(self, image, plotter, show_attributes):
        for obj in self.tracked_obj:
            image = obj.plot_features(image, plotter, show_attributes)
        return image

    def get_attributes(self, names):
        # Check input type
        single_name = False
        if isinstance(names, str):
            names = [names]
            single_name = True

        attributes = {}
        for name in names:
            attribute = []
            for obj in self.tracked_obj:
                attribute.append(obj.get_attributes(name))
            attributes[name] = attribute

        if single_name:
            return attribute
        else:
            return attributes

    def _add_attributes(self):
        for obj in self.tracked_obj:
            if not obj.has_processor():
                obj.attributes += self.processor.attributes
                obj.attributes += self.tracker.attributes
                obj.drawers.append(self.processor.plot_features)
                obj.drawers.append(self.tracker.plot_features)
        return self.tracked_obj

