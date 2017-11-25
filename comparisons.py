import os
import uuid, json

from video import write_segment_to_video


class Comparison:
    def __init__(self, segment1, segment2):
        self.uuid = uuid.uuid4()
        self.left = segment1
        self.right = segment2
        self.label = None
        self.filenames = [None, None]

    # def __iter__(self):
    #     return iter([self.uuid, self.left,self.right,self.label,self.filenames])


class ComparisonsCollector:
    def __init__(self, segments, pretrain_labels, run_number, env):
        self._comparisons = [vars(Comparison(segments[i], segments[i + pretrain_labels]) )for i in range(pretrain_labels)]
        self.run_number = run_number
        self.env = env

    def process_comparisons(self):
        # Create a directory
        req_dir_name = "Human_Preference/static/Collections/run" + str(self.run_number) + "/"
        os.makedirs(os.path.dirname(req_dir_name), exist_ok=True)

        # Cleanup Old files
        for fname in os.listdir(req_dir_name):
            os.remove(os.path.join(req_dir_name, fname))

        path = "Human_Preference/static/Collections/run" + str(self.run_number) + "/"

        print("Generating videos from the segments and storing them in disk.")

        # # Pick every comparison object
        # # For both left and right create 2 filenames, call to save video
        count = 0
        for comparison in self._comparisons:
            left_path = "%s-%s.mp4" % (comparison["uuid"], "left")
            right_path = "%s-%s.mp4" % (comparison["uuid"], "right")

            comparison["filenames"] = [os.path.join("static/Collections/run" + str(self.run_number) + "/", left_path),
                                    os.path.join("static/Collections/run" + str(self.run_number) + "/", right_path)]

            write_segment_to_video(comparison["left"], os.path.join("Human_Preference/",comparison["filenames"][0]), self.env)
            write_segment_to_video(comparison["right"], os.path.join("Human_Preference/",comparison["filenames"][1]), self.env)

            count += 1
            if count % 20 == 0:
                print("Saved {}/{} comparisons".format(count, len(self._comparisons)))

        print("Successfully saved {} comparison videos".format(len(self._comparisons)))

        # Save unlabeled comparisons as a json
        comparison_dict_l = [ {"uuid": str(comp["uuid"]), "left": comp["filenames"][0],
                               "right": comp["filenames"][1], "labels": comp["label"]}
                              for comp in self._comparisons]

        with open("Human_Preference/static/Collections/comparisons.json", "w") as fp:
            json.dump(comparison_dict_l, fp, indent=4)

        print('created json')

    def collect_comparison_labels(self):
        with open("Human_Preference/static/Collections/comparisons.json",'r') as fp:
            labeled_comparisons = json.load(fp)
            # print(labeled_comparisons)
        # Apply these labels on our comparisons object
        for i in range(len(self._comparisons)):
            self._comparisons[i]['label'] = int(labeled_comparisons[i]["labels"])
            
        # from operator import attrgetter
        # x=list(map(attrgetter, self._comparisons))
        return self._comparisons
        # return x
        


