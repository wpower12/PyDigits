from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import mnist
import digitreader

EPOCHS = 5
MNIST_DIR = 'python-mnist/data/'
HIDDEN_LAYER_SIZE = 200
MOMENTUM = 0.1
WEIGHT_DECAY = 0.01

training_data, test_data = mnist.makeMnistDataSets(MNIST_DIR)

print "Number of training images: ", len(training_data)
print "Number of testing images: ", len(test_data)

# Build Network 784 x 50 x 10
fnn = buildNetwork( training_data.indim,
                    HIDDEN_LAYER_SIZE,
                    training_data.outdim,
                    outclass=SoftmaxLayer )

# Building a trainer, Back Propagation
trainer = BackpropTrainer(fnn,
                    dataset=training_data,
                    momentum=MOMENTUM,
                    verbose=True,
                    weightdecay=WEIGHT_DECAY)

# Train
for i in range(EPOCHS):
    trainer.trainEpochs(1)
    train_result = percentError( trainer.testOnClassData(),
                                 training_data['class'])
    test_result = percentError( trainer.testOnClassData(dataset=test_data),
                                 test_data['class'])

    print "Error after epoch ", i
    print "\ttrain e: ", train_result
    print "\ttest  e: ", test_result

# Test on my data set
my_digits = digitreader.read()
for d in range(10):
    # print my_digits[d]
    r = fnn.activate(my_digits[d])
    l = r.tolist()
    max_v = max(l)
    max_i = l.index(max_v)
    print "Evaluating digit image ", d, " output: ", max_i
