class Encoder:

    def __init__(self, pattern, message, permutation, length, frequency, strength, block_number, image_size):
        self.pattern = pattern
        self.message = message
        self.permutation = permutation
        self.length = length
        self.frequency = frequency
        self.strength = strength
        self.block_number = block_number
        self.image_size = image_size

    def show_properties(self):
        return vars(self)


class Decoder(Encoder):

    def __init__(self, pattern, message, permutation, length, frequency, strength, block_number, image_size, channel):
        super().__init__(pattern, message, permutation, length, frequency, strength, block_number, image_size)
        self.channel = channel

    def show_properties(self):
        return vars(self)


class ImagePostproc(Decoder):

    def __init__(self, pattern, message, permutation, block_number, image_size):
        super().__init__(pattern, message, permutation)
        self.block_number = block_number
        self.image_size = image_size


### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

# https://realpython.com/inheritance-composition-python/

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###

### ### ###
### ### ###
### ### ###