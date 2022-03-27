'''Module with supportive functionality'''

# note, comments start with hashtag
# some linter check can be locally disabled


class SupportiveClass: # pylint: disable=too-few-public-methods
    '''Supportive class, can be very supportive
    Sample usage:
    >>> support = SupportiveClass('you can do it')
    >>> support.say()
    'Supportive class says: you can do it'
    '''

    def __init__(self, message):
        self.message = message

    def say(self):
        '''share some supportive toughts'''
        return f'Supportive class says: {self.message}'


def supportive_function(message):
    '''functions can also provide supportive messages'''
    return f'Supportive function says: {message}'
