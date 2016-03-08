/*
 
-0.5518619
-0.88739747
0.5563102
0.83015287
0.30061278
0.24375454
-0.30273616
-0.8389859
-0.5713474  
  
  
  
  
  
 */



public class BackPropLite {
	
	private static double input_layer[];
	private static double hidden_layer[];
	private static double output_layer[];
	
	private static double hidden_layer_error[];
	private static double output_layer_error[];
	
	private static double weights[];
	
	private static double testWeights[] = {-0.5518619, -0.88739747, 0.5563102, 0.83015287, 0.30061278, 0.24375454, -0.30273616, -0.8389859, -0.5713474};
	private static double deltas[];
	
	private static final double LEARNING_RATE = 0.45f;
	private static final double MOMENTUM = 0.7f;
	
	private static final int NUM_INPUT = 2;
	private static final int NUM_HIDDEN = 2;
	private static final int NUM_OUTPUT = 1;
	
	// Training data for XOR problem - excluding BIAS term 
	private static final double[][] train_set = {{1,1,1,1,0},
												 {2,1,0,1,1},
												 {3,0,0,1,0},
												 {4,0,1,1,1}};
	private static final int NUM_PATTERNS = 4;
	
	private static final int NUM_EPOCHS = 1000;
	
	private static double getRandomNumber(int low, int high){
        return ((high - low) * Math.random() + low);
    }
	
	public static double sigmoid(double x){
		return (1.0/(1.0 + Math.exp(-x)));
	}
	
	/*public static double sigmoid(double exponent){
		 return (1.0/(1+Math.pow(Math.E,(-1)*exponent)));
	}*/
	
	
	public static void forwardPass(int pattern_index){
		
		// Load the input layer - skip pattern no. stop at output values
		for(int i = 0; i < NUM_INPUT + 1; i++){
			input_layer[i] = train_set[pattern_index][i+1];
		}
		//DEBUG
		//System.out.println("Input layer");
		//for(int i = 0; i < NUM_INPUT + 1; i++){
		//	System.out.println(input_layer[i]);
		//}
				
		// Calculate values for hidden layer nodes - weighted input -> sigmoid function
		int weight_index = 0;
		for(int i = 0; i < NUM_HIDDEN; i++){
			
			double weighted_sum = 0;
			for(int j = 0; j < NUM_INPUT+1; j++){
				//System.out.println("Input " + input_layer[j] + " * weight " + weights[weight_index] + " = " + (input_layer[j] * weights[weight_index]));
				weighted_sum = weighted_sum + (input_layer[j] * weights[weight_index]);
				weight_index++;
			}
			// Run through sigmoid
			hidden_layer[i] = sigmoid(weighted_sum);
			//System.out.println("Hidden Layer " + hidden_layer[i]);
		}
		
		// Calculate values for output layer nodes - weighted input -> sigmoid function
		for(int i = 0; i < NUM_OUTPUT; i++){
			
			double weighted_sum = 0;
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				//System.out.println("Output " + hidden_layer[j] + " * weight " + weights[weight_index] + " = " + (hidden_layer[j] * weights[weight_index]));
				weighted_sum = weighted_sum + (hidden_layer[j] * weights[weight_index]);
				weight_index++;
			}
			// Run through sigmoid
			output_layer[i] = sigmoid(weighted_sum);
			//System.out.println("Output Layer " + output_layer[i]);
		}
		/*System.out.println("Output layer values");
		for(int z = 0; z < NUM_OUTPUT; z++){
			System.out.print("Output for Pattern " + pattern_index + " Output ");
			
			System.out.format("%.3f%n", output_layer[z]);
			
			System.out.println(" DESIRED " + train_set[pattern_index][NUM_INPUT + 2]);
		}*/
		
	}
	
	public static void calculateOutputErrors(int pattern_index){
		
		double DESIRED;
		for(int i = 0; i < NUM_OUTPUT; i++){
			// First value is pattern index, next values are inputs & bias and last values are desired outputs
			// Offset to desired outputs will be NUM_INPUT + Output Node index
			DESIRED = train_set[pattern_index][(NUM_INPUT + 2) + i];
			
			output_layer_error[i] = (output_layer[i] * (1.0 - output_layer[i]) * (DESIRED - output_layer[i]));
			
			//DEBUG
			//System.out.println("DESIRED " + DESIRED + " Output layer ERROR = " + output_layer_error[i]);
		}
		//DEBUG
		//System.out.println("Output layer errors");
		//for(int i = 0; i < NUM_OUTPUT; i++){
		//	System.out.println(output_layer_error[i] + " ");
		//}
		//System.out.println("");
	}
	
	public static void calculateHiddenErrors(){
		int offset;
		double weighted_error_sum;
		for(int i = 0; i < NUM_HIDDEN; i++){
			weighted_error_sum = 0;
			offset = ((NUM_INPUT + 1) * NUM_HIDDEN) + i;
			//DEBUG
			//System.out.println("Weight[offset] = " + weights[offset]);
			for(int j = 0; j < NUM_OUTPUT; j++){
				//System.out.println("Output Layer Error = " + output_layer_error[j] + " Weight " + weights[offset]);
				weighted_error_sum = weighted_error_sum + (output_layer_error[j] * weights[offset]);
				offset = offset + (NUM_HIDDEN + 1);
			}
			//DEBUG
			//System.out.println("Weighted sum = " + weighted_error_sum);
			hidden_layer_error[i] = (hidden_layer[i] * (1.0 - hidden_layer[i]) * weighted_error_sum);
		}
		//DEBUG
		//System.out.println("Hidden layer errors");
		//for(int i = 0; i < NUM_HIDDEN; i++){
		//	System.out.println(hidden_layer_error[i] + " ");
		//}
		//System.out.println("");
		
	}
	
	public static void updateWeights(){
		int weight_index = 0;
		double delta = 0;
		
		// Hidden layer weights
		for(int i = 0; i < NUM_HIDDEN; i++){
			for(int j = 0; j < NUM_INPUT+1; j++){
				delta = LEARNING_RATE * hidden_layer_error[i] * input_layer[j];
				
				//System.out.println("Delta for weight index " + weight_index + " " + weights[weight_index] + " = " + delta);
				
				weights[weight_index] = weights[weight_index] + delta + (MOMENTUM * deltas[weight_index]);
				deltas[weight_index] = delta;
				weight_index++;
			}
		}
		
		// Output layer weights
		for(int i = 0; i < NUM_OUTPUT; i++){
			for(int j = 0; j < NUM_HIDDEN+1; j++){
				delta = LEARNING_RATE * output_layer_error[i] * hidden_layer[j];
				
				//System.out.println("Delta for weight index " + weight_index + " " + weights[weight_index] + " = " + delta);
				
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
		
		System.out.println("Number of weights = " + weights.length);
		
		// Set the weights to small random values between -1 and 1
		for(int i = 0; i < weights.length; i++){
			//weights[i] = getRandomNumber(-1, 1);
			weights[i] = testWeights[i];
			deltas[i] = 0;
			System.out.println(weights[i]);
		}
		
		// Hard code the BIAS values on the input and hidden layer : T H I S   S U C K S
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
				
				// Output the new weights
				//for(int l = 0; l < weights.length; l++){
				//	System.out.println(weights[l]);
				//}
				
			}
							
			phase++;
		}
		
		for(int i = 0; i < NUM_PATTERNS; i++){
			forwardPass(i);
			for(int z = 0; z < NUM_OUTPUT; z++){
				System.out.print("Output for Pattern " + i + " Output ");
			
				System.out.format("%.3f%n", output_layer[z]);
			
				System.out.println(" DESIRED " + train_set[i][NUM_INPUT + 2]);
			}
			
		}
		
	}

}
