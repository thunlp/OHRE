import json

all_train_data = json.load(open("./hierarchy_fewrel80_train.json"))

rel2sentences = dict()

for instance in all_train_data:
    if instance['relid'] not in rel2sentences.keys():
        rel2sentences[instance['relid']] = []
    rel2sentences[instance['relid']].append(instance)

instance_num_for_test = 100

train_600_instances = []
train_100_instances = []

for relid, instances in rel2sentences.items():
    train_600_instances.extend(instances[:600])
    train_100_instances.extend(instances[600:])


json.dump(train_600_instances, open("./hierarchy_fewrel80_train_600.json", "w"), indent=4)
json.dump(train_100_instances, open("./hierarchy_fewrel80_train_100.json", "w"), indent=4)

