using CharacterClassificationLibrary;
using ConsoleTables;

while (true)
{
    Console.Clear();
    Console.WriteLine("Digit Classification Assignment \n \n");
    Console.WriteLine("Note: All the data samples from all 4 files are merged together and used as a single dataset. 30% of the data is used for testing the network performance and the rest is used for training the network. Data is shuffled each time the network is trained. \n");
    Console.Write("Press any key to start training the network: \n");
    Console.ReadKey();

    (var inputs, var targets) = GetAllData();
    shuffleData(inputs, targets);

    int samplesCount = inputs.Length;
    double testDataRatio = 0.3;
    int testDataCount = (int)(samplesCount * testDataRatio);
    int trainingDataCount = samplesCount - testDataCount;

    int[][,] testInputs = new int[testDataCount][,];
    int[][,] trainingInputs = new int[trainingDataCount][,];
    int[] testTargets = new int[testDataCount];
    int[] trainingTargets = new int[trainingDataCount];

    for (int i = 0; i < testDataCount; i++)
    {
        testInputs[i] = inputs[i];
        testTargets[i] = targets[i];
    }

    for (int i = 0; i < trainingDataCount; i++)
    {
        trainingInputs[i] = inputs[testDataCount + i];
        trainingTargets[i] = targets[testDataCount + i];
    }

    ConvNet classifer = new(trainingInputs, trainingTargets);
    Console.WriteLine("Training started. This may take a few minutes. plesase wait... (500 epochs)");
    classifer.Train();

    double[] predictions = new double[testTargets.Length];
    int correctPredictions = 0;

    for (int i = 0; i < testInputs.GetLength(0); i++)
    {
        var prediction = classifer.Classify(testInputs[i]);
        predictions[i] = prediction;
        var target = targets[i];
        Console.Write($"{i + 1}. prediction: {prediction}, answer: {target} \t");
        if (prediction == target)
        {
            correctPredictions++;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓");
            Console.ForegroundColor = ConsoleColor.White;
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("x");
            Console.ForegroundColor = ConsoleColor.White;
        }
    }
    Console.WriteLine("--------------------------------------------------------------------");
    Console.WriteLine($"correct predictions: {correctPredictions}/{testTargets.Length} \n");

    var confusionMatrix = ConstructConfusionMatrix(testTargets, predictions);
    PrintConfusionMatrix(confusionMatrix);

    var evaluationMatrix = Evaluate(confusionMatrix);
    PrintEvaluationTable(evaluationMatrix);
    PrintMacroScores(evaluationMatrix);

    Console.Write("Press any key to retrain the network: ");
    Console.ReadKey();
}


static void shuffleData(int[][,] inputs, int[] targets)
{
    Random r = new();

    int[] shuffleIndexes = new int[inputs.Length];
    for (int i = 0; i < shuffleIndexes.Length; i++)
    {
        shuffleIndexes[i] = i;
    }
    r.Shuffle(shuffleIndexes);

    var tempInputs = inputs;
    var tempTargets = targets;

    for (int i = 0; i < shuffleIndexes.Length; i++)
    {
        var shuffleIndex = shuffleIndexes[i];
        inputs[i] = tempInputs[shuffleIndex];
        targets[i] = tempTargets[shuffleIndex];
    }
}
static (int[][,] inputs, int[] targets) GetAllData()
{
    (var hwd1Inputs, var hwd1Targets) = ImportData("Data/HWD1");
    (var hwd2Inputs, var hwd2Targets) = ImportData("Data/HWD2");
    (var hwd3Inputs, var hwd3Targets) = ImportData("Data/HWD3");
    (var hwd4Inputs, var hwd4Targets) = ImportData("Data/HWD4");

    int[][,] inputs = new int[hwd1Inputs.Length + hwd2Inputs.Length + hwd3Inputs.Length + hwd4Inputs.Length][,];
    int[] targets = new int[hwd1Targets.Length + hwd2Targets.Length + hwd3Targets.Length + hwd4Targets.Length];

    for (int i = 0; i < inputs.Length; i++)
    {
        inputs[i] = new int[32, 32];
    }

    Array.Copy(hwd1Inputs, inputs, hwd1Inputs.Length);
    Array.Copy(hwd2Inputs, 0, inputs, hwd1Inputs.Length, hwd2Inputs.Length);
    Array.Copy(hwd3Inputs, 0, inputs, hwd1Inputs.Length + hwd2Inputs.Length, hwd3Inputs.Length);
    Array.Copy(hwd2Inputs, 0, inputs, hwd1Inputs.Length + hwd2Inputs.Length + hwd3Inputs.Length, hwd4Inputs.Length);

    Array.Copy(hwd1Targets, targets, hwd1Targets.Length);
    Array.Copy(hwd2Targets, 0, targets, hwd1Targets.Length, hwd2Targets.Length);
    Array.Copy(hwd3Targets, 0, targets, hwd1Targets.Length + hwd2Targets.Length, hwd3Targets.Length);
    Array.Copy(hwd4Targets, 0, targets, hwd1Targets.Length + hwd2Targets.Length + hwd3Targets.Length, hwd4Targets.Length);

    return (inputs, targets);
}
static (int[][,] inputs, int[] targets) ImportData(string path)
{
    int fileLines = CountFileLines(path);
    int sampleCount = fileLines / 33;

    int[][,] inputs = new int[sampleCount][,];
    for (int i = 0; i < sampleCount; i++)
    {
        inputs[i] = new int[32, 32];
    }

    int[] targets = new int[sampleCount];

    using (StreamReader sr = new(path))
    {
        int lineIndex = 0;

        while (true)
        {
            string? currentLine = sr.ReadLine()?.Trim();

            if (currentLine == null)
                break;

            if (lineIndex % 33 != 32)
            {
                for (int col = 0; col < 32; col++)
                {
                    inputs[lineIndex / 33][lineIndex % 33, col] = Convert.ToInt32(currentLine[col] - 48);
                }
            }
            else
            {
                targets[lineIndex / 33] = Convert.ToInt32(currentLine[0] - 48);
            }
            lineIndex++;
        }
    }

    return (inputs, targets);
}
static int CountFileLines(string path)
{
    using (StreamReader r = new StreamReader(path))
    {
        int i = 0;
        while (r.ReadLine() != null) { i++; }
        return i;
    }
}
static int[,] ConstructConfusionMatrix(int[] testTargets, double[] predictions)
{
    var confusionMatrix = new int[10, 10];

    for (int i = 0; i < testTargets.Length; i++)
    {
        var target = testTargets[i];
        var prediction = (int)predictions[i];

        confusionMatrix[target, prediction]++;
    }

    return confusionMatrix;
}
static void PrintConfusionMatrix(int[,] confusionMatrix)
{
    ConsoleTable table = new("actual/prediction", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9");

    for (int i = 0; i < confusionMatrix.GetLength(0); i++)
    {
        table.AddRow(i, confusionMatrix[i, 0], confusionMatrix[i, 1], confusionMatrix[i, 2], confusionMatrix[i, 3], confusionMatrix[i, 4], confusionMatrix[i, 5], confusionMatrix[i, 6], confusionMatrix[i, 7], confusionMatrix[i, 8], confusionMatrix[i, 9]);
    }

    table.Configure((x) => x.EnableCount = false);
    Console.WriteLine("Confusion matrix:");
    table.Write();
}
static double[,] Evaluate(int[,] confusionMatrix)
{
    double[,] output = new double[confusionMatrix.GetLength(0), 4];

    int totalCount = 0;
    for (int i = 0; i < confusionMatrix.GetLength(0); i++)
    {
        for (int j = 0; j < confusionMatrix.GetLength(1); j++)
        {
            totalCount += confusionMatrix[i, j];
        }
    }

    for (int i = 0; i < confusionMatrix.GetLength(0); i++)
    {
        double tpCount = 0;
        double tnCount = 0;
        double fpCount = 0;
        double fnCount = 0;

        int tpIndex = i;
        tpCount = confusionMatrix[i, tpIndex];

        for (int j = 0; j < confusionMatrix.GetLength(1); j++)
        {
            fpCount = j == i ? fpCount : fpCount + confusionMatrix[i, j];
        }

        for (int j = 0; j < confusionMatrix.GetLength(0); j++)
        {
            fnCount = j == i ? fnCount : fnCount + confusionMatrix[j, i];
        }

        tnCount = totalCount - tpCount - fnCount - fpCount;
        output[i, 0] = tpCount / (tpCount + fpCount);
        output[i, 1] = (tpCount + fnCount == 0) ? 0 : tpCount / (tpCount + fnCount);
        output[i, 2] = (tpCount + tnCount) / totalCount;
        output[i, 3] = (output[i, 0] + output[i, 1] == 0) ? 0 : (2 * output[i, 0] * output[i, 1]) / (output[i, 0] + output[i, 1]);
    }
    return output;
}
static void PrintEvaluationTable(double[,] evaluationMatrix)
{
    ConsoleTable table = new("", "Precision", "Recall", "Accuracy", "F1 Score");

    for (int i = 0; i < evaluationMatrix.GetLength(0); i++)
    {
        var precision = (evaluationMatrix[i, 0] * 100).ToString("F2") + "%";
        var recall = (evaluationMatrix[i, 1] * 100).ToString("F2") + "%";
        var accuracy = (evaluationMatrix[i, 2] * 100).ToString("F2") + "%";
        var f1Score = (evaluationMatrix[i, 3] * 100).ToString("F2") + "%";
        table.AddRow(i, precision, recall, accuracy, f1Score);
    }

    table.Configure((x) => x.EnableCount = false);

    Console.WriteLine("Evaluation table:");
    table.Write();
}
static void PrintMacroScores(double[,] evaluationMatrix)
{
    double[] MacroScores = new double[evaluationMatrix.GetLength(1)];
    for (int i = 0; i < evaluationMatrix.GetLength(1); i++)
    {
        double weightedAverage = 0;
        for (int j = 0; j < evaluationMatrix.GetLength(0); j++)
        {
            weightedAverage += evaluationMatrix[j, i];
        }
        weightedAverage /= evaluationMatrix.GetLength(0);
        MacroScores[i] = weightedAverage;
    }

    var macroPrecision = (MacroScores[0] * 100).ToString("F2") + "%";
    var macroRecall = (MacroScores[1] * 100).ToString("F2") + "%";
    var macroAccuracy = (MacroScores[2] * 100).ToString("F2") + "%";
    var macroF1Score = (MacroScores[3] * 100).ToString("F2") + "%";

    Console.WriteLine($"Macro Precision: {macroPrecision}");
    Console.WriteLine($"Macro Recall: {macroRecall}");
    Console.WriteLine($"Macro Accuracy: {macroAccuracy}");
    Console.WriteLine($"Macro F1 Score: {macroF1Score}");
}