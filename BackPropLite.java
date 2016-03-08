// Author: Stephen Sheridan
// Created: 16/05/2014

public class BackPropLite {
	
	private static double input_layer[];
	private static double hidden_layer[];
	private static double output_layer[];
	private static double hidden_layer_error[];
	private static double output_layer_error[];
	private static double weights[];
	
	// For debugging purposes - fix the weights to make sure all the calculations are correct
	//private static double testWeights[] = {-0.5518619, -0.88739747, 0.5563102, 0.83015287, 0.30061278, 0.24375454, -0.30273616, -0.8389859, -0.5713474};
	private static double deltas[];
	
	private static final double LEARNING_RATE = 0.45f;
	private static final double MOMENTUM = 0.7f;
	private static final int NUM_INPUT = 2;
	private static final int NUM_HIDDEN = 2;
	private static final int NUM_OUTPUT = 1;
	private static final int NUM_PATTERNS = 4;
	private static final int NUM_EPOCHS = 1500;
	
	// Training data for XOR problem - excluding BIAS term 
	private static final double[][] train_set = {{1,1,1,1,0},
												 {2,1,0,1,1},
												 {3,0,0,1,0},
												 {4,0,1,1,1}};
	
	
	private static double getRandomNumber(int low, int high){
        return ((high - low) * Math.random() + low);
    }
	
	public static double sigmoid(double x){
		return (1.0/(1.0 + Math.exp(-x)));
	}
	
	public static void forwardPass(int pattern_index){
		// Load the input layer - skip pattern no. stop at output values
		for(int i = 0; i < NUM_INPUT + 1; i++){
			input_layer[i] = train_set[pattern_index][i+1];
		}
		// Calculate values for hidden layer nodes - weighted input -> sigmoid function
		int weight_index = 0;
		for(int i = 0; i < NUM_HIDDEN; i++){
			double weighted_sum = 0;
			for(int j = 0; j < NUM_INPUT+1; j++){
				weighted_sum = weighted_sum + (input_layer[j] * weights[weight_index]);
				weight_index++;
			}
			// Run through sigmoid
			hidden_layer[i] = sigmoid(weighted_sum);
		}
		
		// Calculate values for output layer nodes - weighted input -> sigmoid function
		for(int i = 0; i < NUM_OUTPUT; i++){
			double weighted_sum = 0;
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				weighted_sum = weighted_sum + (hidden_layer[j] * weights[weight_index]);
				weight_index++;
			}
			// Run through sigmoid
			output_layer[i] = sigmoid(weighted_sum);
		}
		
	}
	
	public static void calculateOutputErrors(int pattern_index){
		double DESIRED;
		for(int i = 0; i < NUM_OUTPUT; i++){
			// First value is pattern index, next values are inputs & bias and last values are desired outputs
			// Offset to desired outputs will be NUM_INPUT + Output Node index
			DESIRED = train_set[pattern_index][(NUM_INPUT + 2) + i];
			output_layer_error[i] = (output_layer[i] * (1.0 - output_layer[i]) * (DESIRED - output_layer[i]));
		}
	}
	
	public static void calculateHiddenErrors(){
		int offset;
		double weighted_error_sum;
		for(int i = 0; i < NUM_HIDDEN; i++){
			weighted_error_sum = 0;
			offset = ((NUM_INPUT + 1) * NUM_HIDDEN) + i;
			for(int j = 0; j < NUM_OUTPUT; j++){
				weighted_error_sum = weighted_error_sum + (output_layer_error[j] * weights[offset]);
				offset = offset + (NUM_HIDDEN + 1);
			}
			hidden_layer_error[i] = (hidden_layer[i] * (1.0 - hidden_layer[i]) * weighted_error_sum);
		}
	}
	
	public static void updateWeights(){
		int weight_index = 0;
		double delta = 0;
		// Hidden layer weights
		for(int i = 0; i < NUM_HIDDEN; i++){
			for(int j = 0; j < NUM_INPUT+1; j++){
				delta = LEARNING_RATE * hidden_layer_error[i] * input_layer[j];
				weights[weight_index] = weights[weight_index] + delta + (MOMENTUM * deltas[weight_index]);
				deltas[weight_index] = delta;
				weight_index++;
			}
		}
		
		// Output layer weights
		for(int i = 0; i < NUM_OUTPUT; i++){
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				delta = LEARNING_RATE * output_layer_error[i] * hidden_layer[j];
				weights[weight_index] = weights[weight_index] + delta + (MOMENTUM * deltas[weight_index]);
				deltas[weight_index] = delta;
				weight_index++;
			}
		}
	}
	
	public static void main(String[] args){
		
		// Create the network
		input_layer = new double[NUM_INPUT + 1];	// +1 for BIAS
		hidden_layer = new double[NUM_HIDDEN + 1];	// +1 for BIAS
		output_layer = new double[NUM_OUTPUT];
		
		hidden_layer_error = new double[NUM_HIDDEN];
		output_layer_error = new double[NUM_OUTPUT];
		
		// Three layer network requires two BIAS values
		// Assume BIAS weights are at end of each layers weights
		weights = new double[(NUM_HIDDEN * NUM_INPUT)+(NUM_HIDDEN * NUM_OUTPUT) + NUM_HIDDEN + NUM_OUTPUT];
		deltas = new double[weights.length];
		
		// Set the weights to small random values between -1 and 1
		for(int i = 0; i < weights.length; i++){
			weights[i] = getRandomNumber(-1, 1);
			//weights[i] = testWeights[i];
			deltas[i] = 0;
		}
		
		// Hard code the BIAS values on the input and hidden layer : T H I S   S U C K S
		// Need to implement a better way to do this, works for now though
		input_layer[NUM_INPUT ] = 1;
		hidden_layer[NUM_HIDDEN] = 1;
		
		// Train the network
		int phase = 0;
		while(phase < NUM_EPOCHS){
			for(int i = 0; i < NUM_PATTERNS; i++){
				forwardPass(i);
				calculateOutputErrors(i);	// Calculate error values for output layer nodes
				calculateHiddenErrors();	// Calculate error values for hidden layer nodes
				updateWeights();
			}
			phase++;
		}
		
		// Output the final values
		for(int i = 0; i < NUM_PATTERNS; i++){
			forwardPass(i);
			for(int z = 0; z < NUM_OUTPUT; z++){
				System.out.print("Output for Pattern " + i + " ACTUAL ");
				System.out.format("%.3f DESIRED %.2f \n", output_layer[z], train_set[i][NUM_INPUT + 2]);
			}
			
		}
		
	}

}
