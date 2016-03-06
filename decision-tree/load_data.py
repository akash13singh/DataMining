from openml.apiconnector import APIConnector
apikey = '6bf2b9754012ef63cb8473a41e8e37bb'
def load(dataset_id):
    print 'Loadding data_id %d' % (dataset_id)
    connector = APIConnector(apikey=apikey)
    dataset = connector.download_dataset(dataset_id)
    return dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
