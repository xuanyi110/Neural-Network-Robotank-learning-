package JBot_NN;
import java.lang.Math;

import robocode.RobocodeFileOutputStream;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;

public class NeuralNet implements NeuralNetInterface
{
	// ------------------------hyper-parameters------------------------
    private int argNumInputs;
    private int argNumHidden;
    private double argLearningRate;
    private double argMomentumTerm;
    private int argA;
    private int argB;
	public static final double learningRate = 0.3;
	public static final double discountFactor = 0.9;

    // ------------------------parameters------------------------
    private double[][] layer1;
    private double[] layer2;
    private double[] hiddenBias;
    private double outputBias;
    
    // ------------------------Global Variables------------------------
    private double[][] delta1;
    private double[] delta2;
    private double[] delataHiddenBias;
    private double deltaOutputBias;

    private double[] hiddenOutput;
    private double[] hiddenSum;
    private double outputSum;

    public NeuralNet(int argNumInputs,
                     int argNumHidden,
                     double argLearningRate,
                     double argMomentumTerm,
                     int argA,
                     int argB)
    {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argLearningRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.argA = argA;
        this.argB = argB;

        this.layer1 = new double[argNumInputs][argNumHidden];
        this.layer2 = new double[argNumHidden];
        this.hiddenBias = new double[argNumHidden];
        this.outputBias = 0;
        this.delta1 = new double[argNumInputs][argNumHidden];
        this.delta2 = new double[argNumHidden];
        this.delataHiddenBias = new double[argNumHidden];
        this.deltaOutputBias = 0;
        this.hiddenOutput = new double[argNumHidden];
        this.hiddenSum = new double[argNumHidden];
        this.outputSum = 0;
        
        this.initializeWeights();
    }

    public double sigmoid(double x)
    {
        return 2.0 / (1 + Math.exp(-x)) - 1;
    }

    public double customSigmoid(double x)
    {
        return (this.argB - this.argA) / (1 + Math.exp(-x)) + this.argA;
    }

    public double sigmoidDerivative(double x)
    {
        return 2.0 * Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)));
    }

    public double customSigmoidDerivative(double x)
    {
        return (this.argB - this.argA) * Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)));
    }

    public void initializeWeights()
    {
        for (int j = 0; j < this.argNumHidden; j++)
        {
            this.layer2[j] = Math.random() - 0.5;
            this.delta2[j] = 0.0;
            this.hiddenBias[j] = Math.random() - 0.5;
            this.delataHiddenBias[j] = 0.0;
            this.hiddenOutput[j] = 0.0;
            this.hiddenSum[j] = 0.0;
            for (int i = 0; i < this.argNumInputs; i++)
            {
                this.layer1[i][j] = Math.random() - 0.5;
                this.delta1[i][j] = 0.0;
            }
        }
        this.outputBias = Math.random() - 0.5;
        this.outputSum = 0;
        this.deltaOutputBias = 0;
    }

    public void zeroWeights()
    {
        for (int j = 0; j < this.argNumHidden; j++)
        {
            layer2[j] = 0.0;
            delta2[j] = 0.0;
            this.hiddenBias[j] = 0.0;
            this.delataHiddenBias[j] = 0.0;
            for (int i = 0; i < this.argNumInputs; i++)
            {
                layer1[i][j] = 0.0;
                delta1[i][j] = 0.0;
            }
        }
        this.outputBias = 0;
        this.deltaOutputBias = 0;
    }

    public void testWeights()  // 0.1
    {
        for (int j = 0; j < this.argNumHidden; j++)
        {
            layer2[j] = 0.1;
            delta2[j] = 0.0;
            this.hiddenBias[j] = 0.1;
            this.delataHiddenBias[j] = 0.0;
            for (int i = 0; i < this.argNumInputs; i++)
            {
                layer1[i][j] = 0.1;
                delta1[i][j] = 0.0;
            }
        }
        this.outputBias = 0.1;
        this.deltaOutputBias = 0.0;
    }

    private void hiddenOutputFor(double [] X)
    {
        for (int j = 0; j < this.argNumHidden; j++)
        {
            this.hiddenSum[j] = 0;
            for (int i = 0; i < this.argNumInputs; i++)
            {
                this.hiddenSum[j] += X[i] * this.layer1[i][j];
            }
            this.hiddenSum[j] += this.hiddenBias[j] * 1.0;
            this.hiddenOutput[j] = customSigmoid(this.hiddenSum[j]);
        }
    }

    public double outputFor(double [] X)
    {
        this.hiddenOutputFor(X);
        this.outputSum = 0;

        for (int j = 0; j < this.argNumHidden; j++)
        {
            this.outputSum += this.hiddenOutput[j] * this.layer2[j];
        }

        this.outputSum += 1.0 * this.outputBias;
        
//        return customSigmoid(this.outputSum);
        return this.outputSum;
    }

    public double train(double [] X, double argValue)
    {
        double predict = outputFor(X);
//        double outputError = (argValue - predict) * customSigmoidDerivative(this.outputSum);
        double outputError = argValue - predict;
        updateLayer2(outputError);
        double[] hiddenErrors = new double[this.argNumHidden];
        for (int j = 0; j < this.argNumHidden; j++)
        {
            hiddenErrors[j] = outputError * this.layer2[j] * customSigmoidDerivative(this.hiddenSum[j]);
        }
        updateLayer1(X, hiddenErrors);
        return argValue - predict;
    }

    private void updateLayer1(double [] X, double [] hiddenErrors)
    {
        double tempDelta;
        for (int j = 0; j < this.argNumHidden; j++)
        {
            tempDelta = this.argMomentumTerm * this.delataHiddenBias[j] + this.argLearningRate * hiddenErrors[j] * 1.0;
            this.hiddenBias[j] += tempDelta;
            this.delataHiddenBias[j] = tempDelta;

            for (int i = 0; i < this.argNumInputs; i++)
            {
                tempDelta = this.argMomentumTerm * this.delta1[i][j] + this.argLearningRate * hiddenErrors[j] * X[i];
                this.layer1[i][j] += tempDelta;
                this.delta1[i][j] = tempDelta;
            }
        }
    }

    private void updateLayer2(double outputError)
    {
        double tempDelta;
        tempDelta = this.argMomentumTerm * this.deltaOutputBias + this.argLearningRate * outputError * 1.0;
        this.outputBias += tempDelta;
        // this.outputBias = 0;
        this.deltaOutputBias = tempDelta;
        for (int j = 0; j < this.argNumHidden; j++)
        {
            tempDelta = this.argMomentumTerm * this.delta2[j] + this.argLearningRate * outputError * this.hiddenOutput[j];
            this.layer2[j] += tempDelta;
            this.delta2[j] = tempDelta;
        }
    }
    
    private double[] getXArray(double[] states, int action) {
		double[] X = new double[10];
    	X[0] = states[0] * 2 - 1;
		X[1] = states[1];
		X[2] = states[2] * 2 - 1;
		X[3] = states[3] * 2 - 1;
		X[4] = states[4] * 2 - 1;

//		X[0] = states[0];
//		X[1] = states[1];
//		X[2] = states[2];
//		X[3] = states[3];
//		X[4] = states[4];
		if (action != 4) {
			X[5] = action / 1.5 - 1;
			X[6] = 0;
		}
		else {
			X[5] = 0;
			X[6] = 1;
		}
		return X;
	}
    
    public double getQ(double[] states, int action) {
//		return outputFor(getXArray(states, action)) * 50 + 30;  // corner
		return outputFor(getXArray(states, action)) + 30;  // corner linear
//		return outputFor(getXArray(states, action)) * 25 + 2;  // fire
//		return outputFor(getXArray(states, action)) + 2;  // fire linear
	}
    
    private double QtoY(double Q) {
//		return (Q - 30) / 50;  // corner
		return (Q - 30);  // corner linear
//		return (Q - 2) / 25;  // fire
//		return (Q - 2);  // fire linear
	}
    
    public void updateNNOff(double[] states_before, double[] states_after, int action, double r) {
    	double newQ = r + discountFactor * maxQFor(states_after);
    	updateQ(states_before, action, newQ);
	}
    
    public void updateQ(double[] states, int action, double newQ) {
//    	System.out.print("old Q: ");
//    	System.out.print(getQ(states, action));
//    	System.out.print(", delta Q: ");
//    	System.out.print(delta);
//    	System.out.print(", new y: ");
//    	System.out.println(QtoY(getQ(states, action) + delta));
		train(getXArray(states, action), QtoY(newQ));
//		train(getXArray(states, action), QtoY(getQ(states, action) + delta));
	}
    
//    public void updateNNOff(double[] states_before, double[] states_after, int action, double r) {
//    	if (r == 0) return;
//		double delta = learningRate * (r + discountFactor * maxQFor(states_after) - getQ(states_before, action));
//		updateQ(states_before, action, delta);
//	}
//    
//    public void updateQ(double[] states, int action, double delta) {
//    	System.out.print("old Q: ");
//    	System.out.print(getQ(states, action));
//    	System.out.print(", delta Q: ");
//    	System.out.print(delta);
//    	System.out.print(", new y: ");
//    	System.out.println(QtoY(getQ(states, action) + delta));
//		train(getXArray(states, action), QtoY(getQ(states, action) + delta));
////		train(getXArray(states, action), QtoY(getQ(states, action) + delta));
//	}
    
    public int pickAction(double[] states, boolean rand) {
		if (rand) return (int)Math.random() * Action.numActions;
		else {
			return pickBestAction(states);
		}
	}
    
    public int pickBestAction(double[] states) {
    	double maxQ = getQ(states, 4);
    	int bestAction = 4;
		for (int i = 0; i < 4; i++) {
			double Q = getQ(states, i);
			if (Q > maxQ) {
				maxQ = Q;
				bestAction = i;
			}
		}
		return bestAction;
	}
    
    public double maxQFor(double[] states) {
		double maxQ = getQ(states, 4);
		for (int i = 0; i < 4; i++) {
			double Q = getQ(states, i);
			if (Q > maxQ) maxQ = Q;
		}
		return maxQ;
	}

    public void printLayers(int mode)
    {
        if (mode == 0 || mode == 1)
        {
            for (int i = 0; i < this.argNumInputs; i++)
            {
                for (int j = 0; j < this.argNumHidden; j++)
                {
                    System.out.print(this.layer1[i][j]);
                    System.out.print(' ');
                }
                System.out.println();
            }
        }
        if (mode == 0 || mode == 2)
        {
            for (int j = 0; j < this.argNumHidden; j++)
            {
                System.out.print(this.layer2[j]);
                System.out.print(' ');
            }
            System.out.println();
        }
    }

    public void save(File argFile)
    {
    	PrintStream write = null; 
		try { 
			write = new PrintStream(new RobocodeFileOutputStream(argFile)); 
			for (int i = 0; i < this.argNumInputs; i++)
				for (int j = 0; j < this.argNumHidden; j++)
					write.println(new Double(this.layer1[i][j]));
			for (int i = 0; i < this.argNumHidden; i++)
				write.println(new Double(this.layer2[i]));
			for (int i = 0; i < this.argNumHidden; i++)
				write.println(new Double(this.hiddenBias[i]));
			write.println(new Double(this.outputBias));
			if (write.checkError()) 
				System.out.println("Could not save the data!"); 
			write.close(); 
		} 
		catch (IOException e) { 
	   //   System.out.println("IOException trying to write: " + e); 
		} 
		finally { 
			try { 
				if (write != null) 
					write.close(); 
			} 
			catch (Exception e) { 
	  //      System.out.println("Exception trying to close witer: " + e); 
			} 
		}
    }
    
    public void loadFile(File file) throws IOException {
		BufferedReader read = null;
	    try {
	    	read = new BufferedReader(new FileReader(file));
	    	for (int i = 0; i < this.argNumInputs; i++)
				for (int j = 0; j < this.argNumHidden; j++)
					this.layer1[i][j] = Double.parseDouble(read.readLine());
			for (int i = 0; i < this.argNumHidden; i++)
				this.layer2[i] = Double.parseDouble(read.readLine());
			for (int i = 0; i < this.argNumHidden; i++)
				this.hiddenBias[i] = Double.parseDouble(read.readLine());
			this.outputBias = Double.parseDouble(read.readLine());
	    }
	    catch (IOException e) {
	    //  System.out.println("IOException trying to open reader: " + e);
	    	initializeWeights();
	    }
	    catch (NumberFormatException e) {
	    	initializeWeights();
	    }
	    finally {
	    	try {
	    		if (read != null)
	    			read.close();
	    	}
	    	catch (IOException e) {
	     //   System.out.println("IOException trying to close reader: " + e);
	    	}
	    }
	}

    public void load(String argFileName) throws IOException
    {}
}