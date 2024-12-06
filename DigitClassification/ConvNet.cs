using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CharacterClassificationLibrary
{
    public class ConvNet
    {
        private Sequential Model;
        private Tensor Features;
        private Tensor Targets;

        public ConvNet(int[][,] inputs, int[] targets)
        {
            int[,,] threeDimensionalInputs = new int[inputs.Length, 32, 32];
            for (int i = 0; i < threeDimensionalInputs.GetLength(0); i++)
            {
                for (int j = 0; j < threeDimensionalInputs.GetLength(1); j++)
                {
                    for (int k = 0; k < threeDimensionalInputs.GetLength(2); k++)
                    {
                        threeDimensionalInputs[i, j, k] = inputs[i][j, k];
                    }
                }
            }
            Features = tensor(threeDimensionalInputs).reshape(threeDimensionalInputs.GetLength(0), 1, 32, 32).to(float32);
            Targets = tensor(targets).to(int64);

            Model = nn.Sequential(
                                ("conv1", nn.Conv2d(1, 10, 3)),
                                ("relu1", nn.ReLU()),
                                ("maxpool1", nn.MaxPool2d(2, 2)),
                                ("flatten", nn.Flatten()),
                                ("linear2", nn.Linear(10 * 15 * 15, 10))
                                );
        }

        public void Train()
        {
            var optimizer = optim.Adam(Model.parameters());
            var lossFn = nn.CrossEntropyLoss();

            int epochs = 500;
            Console.WriteLine("---------------------------------------");
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                optimizer.zero_grad();
                var output = Model.forward(Features);
                var loss = lossFn.forward(output, Targets);
                loss.backward();
                optimizer.step();
                Console.WriteLine($"Epoch {epoch} ended.");
            }
            Console.WriteLine("---------------------------------------");
        }

        public double Classify(int[,] input)
        {
            var inputTensor = tensor(input).reshape(1, 1, 32, 32).to(float32);
            var output = Model.forward(inputTensor);
            var prediction = output.argmax(1).item<long>();
            return prediction;
        }
    }
}
