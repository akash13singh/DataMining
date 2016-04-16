from openml import tasks,runs
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xmltodict
from sklearn import ensemble

task = tasks.get_task(14951)
#clf = ensemble.RandomForestClassifier()
#clf = AdaBoostClassifier(algorithm="SAMME.R",n_estimators=700)
#clf = AdaBoostClassifier(algorithm="SAMME",n_estimators=5000)
clf = RandomForestClassifier(warm_start=True,n_estimators=128,criterion="entropy",min_samples_split=20,bootstrap=True,random_state=123)
run = runs.run_task(task, clf)
return_code, response = run.publish()

# get the run id for reference
if(return_code == 200):
    response_dict = xmltodict.parse(response)
    run_id = response_dict['oml:upload_run']['oml:run_id']
    print("Uploaded     run with id %s. Check it at www.openml.org/r/%s" % (run_id,run_id))