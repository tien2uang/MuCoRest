import numpy as np


class StateVector():
    def __init__(self, state):
        self.state_vector = self.preprocess_state(state)
        self.state = state

    def get_state_vector(self):
        return self.state_vector
    def get_state(self):
        return self.state

    def preprocess_state(self, state):
        # Example preprocessing: Extract features from the state
        operation_id = self.encode_operation_id(state['operation_id'])
        method = self.encode_method(state['method'])
        path = self.encode_path(state['path'])
        parameters = self.encode_parameters(state['parameters'])
        responses = self.encode_responses(state['responses'])

        # Concatenate all features into a single numerical array
        state_vector = np.concatenate([operation_id, method, path, parameters, responses])
        return state_vector

    def encode_operation_id(self, operation_id):
        # Example: Use a simple hash or an embedding (here, just a simple encoding)
        return np.array([hash(operation_id) % 1000], dtype=np.float32)

    def encode_method(self, method):
        methods = ['get', 'post', 'put', 'delete', 'patch']
        return np.array([1 if method == m else 0 for m in methods], dtype=np.float32)

    def encode_path(self, path):
        # Example: Use a predefined dictionary or hashing
        return np.array([hash(path) % 1000], dtype=np.float32)

    def encode_parameters(self, parameters):
        # Example: Count parameters, required parameters, and types
        num_params = len(parameters)
        num_required = sum(1 for p in parameters if p.get('required', False))
        types = [p['schema']['type'] for p in parameters if 'schema' in p]
        type_counts = [types.count(t) for t in ['string', 'integer', 'object', 'array', 'boolean']]
        return np.array([num_params, num_required] + type_counts, dtype=np.float32)

    def encode_responses(self, responses):
        # Example: Encode the presence of specific response codes
        response_codes = ['200', '201', '400', '401', '403', '404', '500']
        return np.array([1 if code in responses else 0 for code in response_codes], dtype=np.float32)


if __name__ == '__main__':
    operation = {'operation_id': 'postPersonUsingPOST', 'method': 'post', 'path': '/api/person',
                 'parameters': [
        {'in': 'body', 'name': 'person', 'description': 'person', 'required': True, 'schema': {'type': 'object',
                                                                                               'properties': {
                                                                                                   'address': {
                                                                                                       'type': 'object',
                                                                                                       'properties': {
                                                                                                           'city': {
                                                                                                               'type': 'string'},
                                                                                                           'country': {
                                                                                                               'type': 'string'},
                                                                                                           'number': {
                                                                                                               'type': 'integer',
                                                                                                               'format': 'int32'},
                                                                                                           'postcode': {
                                                                                                               'type': 'string'},
                                                                                                           'street': {
                                                                                                               'type': 'string'}},
                                                                                                       'title': 'Address'},
                                                                                                   'age': {
                                                                                                       'type': 'integer',
                                                                                                       'format': 'int32'},
                                                                                                   'cars': {
                                                                                                       'type': 'array',
                                                                                                       'items': {
                                                                                                           'type': 'object',
                                                                                                           'properties': {
                                                                                                               'brand': {
                                                                                                                   'type': 'string'},
                                                                                                               'maxSpeedKmH': {
                                                                                                                   'type': 'number',
                                                                                                                   'format': 'float'},
                                                                                                               'model': {
                                                                                                                   'type': 'string'}},
                                                                                                           'title': 'Car'}},
                                                                                                   'createdAt': {
                                                                                                       'type': 'string',
                                                                                                       'format': 'date-time'},
                                                                                                   'firstName': {
                                                                                                       'type': 'string'},
                                                                                                   'id': {
                                                                                                       'type': 'object',
                                                                                                       'properties': {
                                                                                                           'timestamp': {
                                                                                                               'type': 'integer',
                                                                                                               'format': 'int32'}},
                                                                                                       'title': 'ObjectIdReq'},
                                                                                                   'insurance': {
                                                                                                       'type': 'boolean'},
                                                                                                   'lastName': {
                                                                                                       'type': 'string'}},
                                                                                               'title': 'PersonReq'}}],
                 'responses': {'201': {'description': 'Created', 'schema': {'type': 'object', 'properties': {
                     'address': {'type': 'object',
                                 'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'},
                                                'number': {'type': 'integer', 'format': 'int32'},
                                                'postcode': {'type': 'string'}, 'street': {'type': 'string'}},
                                 'title': 'Address'}, 'age': {'type': 'integer', 'format': 'int32'},
                     'cars': {'type': 'array', 'items': {'type': 'object', 'properties': {'brand': {'type': 'string'},
                                                                                          'maxSpeedKmH': {
                                                                                              'type': 'number',
                                                                                              'format': 'float'},
                                                                                          'model': {'type': 'string'}},
                                                         'title': 'Car'}},
                     'createdAt': {'type': 'string', 'format': 'date-time'}, 'firstName': {'type': 'string'},
                     'id': {'type': 'object', 'properties': {'date': {'type': 'string', 'format': 'date-time'},
                                                             'timestamp': {'type': 'integer', 'format': 'int32'}},
                            'title': 'ObjectIdRes'}, 'insurance': {'type': 'boolean'}, 'lastName': {'type': 'string'}},
                                                                            'title': 'PersonRes'}},
                               '401': {'description': 'Unauthorized'}, '403': {'description': 'Forbidden'},
                               '404': {'description': 'Not Found'}}}

    state_vector= StateVector(operation)
