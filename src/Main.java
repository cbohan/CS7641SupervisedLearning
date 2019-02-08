import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Main {
	//Where to save the .arff files.
	public static final String POKEMON_TRAINING_DATA_SET = "pokemonTraining.arff"; 
	public static final String POKEMON_TEST_DATA_SET = "pokemonTest.arff"; 
	public static final String CUSTOMER_SATISFACTION_TRAINING_DATA_SET = "customerSatisfactionTraining.arff"; 
	public static final String CUSTOMER_SATISFACTION_TEST_DATA_SET = "customerSatisfactionTest.arff"; 
	
	private static final boolean DO_POKEMON_DATASET = false; //Create and test classifier for pokemon dataset?
	private static final boolean DO_CUSTOMER_SATISFACTION_DATASET = true; //Create and test classifier for customer satisfaction dataset?
	private static final boolean TEST_AGAINST_TEST_SET = true;  //True = test against test dataset, false = test against training dataset.
	private static final int TEST_RUNS = 1; //How many times to generate new data, create a classifier, and test the classifier.
	
	private static final int ENTRIES_PER_POKEMON = 25; //How much data to create for the pokemon dataset (total data = this value * 34).
	private static final int TOTAL_CUSTOMER_ENTRIES = 1000; //How much data to create for the customer satisfaction dataset.
	
	private static final int NEURAL_NET_TRAINING_TIME = 500; 
	private static final int K = 10; //K in KNN
	private static final int BOOSTING_ITERATIONS = 100;
	
	//NOTE: Only one of these should be true at a time when trying to get running time for 
	//a type of classifier.
	private static final boolean DECISION_TREE = false;
	private static final boolean NEURAL_NETWORK= false;
	private static final boolean KNN = false;
	private static final boolean BOOSTING = false;
	private static final boolean SVM = true;
	
	//DO NOT TOUCH FROM HERE ON
	private static Instances pokemonTrainingDataSet;
	private static Instances pokemonTestDataSet;
	private static Instances customerSatisfactionTrainingDataSet;
	private static Instances customerSatisfactionTestDataSet;
	
	private static J48 pokemonDecisionTreeClassifier, customerSatisfactionDecisionTreeClassifier;
	private static MultilayerPerceptron pokemonNeuralNetworkClassifier, customerSatisfactionNeuralNetworkClassifier;
	private static IBk pokemonKNNClassifier, customerSatisfactionKNNClassifier;
	private static AdaBoostM1 pokemonBoostingClassifier, customerSatisfactionBoostingClassifier;
	private static SMO pokemonSVMClassifier, customerSatisfactionSVMClassifier;
	
	public static void main(String[] args) {
		long startTime = System.nanoTime();
		
		for (int i = 0; i < TEST_RUNS; i++) {
			//Used to generate the data files.
			PokemonARFFGenerator.generateFile(ENTRIES_PER_POKEMON, false);
			CustomerSatisfactionARFFGenerator.generateFile(TOTAL_CUSTOMER_ENTRIES, false);
			System.out.println();
			
			//Load the data.
			loadInstances();
			
			//Do training.
			if (DECISION_TREE)
				trainDecisionTree();
			if (NEURAL_NETWORK)
				trainNeuralNetwork();
			if (KNN)
				trainKNN();
			if (BOOSTING)
				trainBoosting();
			if (SVM)
				trainSVM();
			
			//Evaluate the classifiers on the test data.
			if (DECISION_TREE)
				test(pokemonDecisionTreeClassifier, customerSatisfactionDecisionTreeClassifier, "decision tree");
			if (NEURAL_NETWORK)
				test(pokemonNeuralNetworkClassifier, customerSatisfactionNeuralNetworkClassifier, "neural network");
			if (KNN)
				test(pokemonKNNClassifier, customerSatisfactionKNNClassifier, "knn");
			if (BOOSTING)
				test(pokemonBoostingClassifier, customerSatisfactionBoostingClassifier, "boosting");
			if (SVM)
				test(pokemonSVMClassifier, customerSatisfactionSVMClassifier, "svm");
		}
		
		long endTime = System.nanoTime();
		long difference = endTime - startTime;
		double seconds = (double)difference / 1000000000.0;
		double averageTime = seconds / TEST_RUNS;
		
		System.out.println("Average time: " + averageTime);
		
	}
	
	private static void test(Classifier pokemonClassifier, Classifier customerSatisfactionClassifier, String classifierName) {
		Evaluation pokemonEvaluation, customerSatisfactionEvaluation;
		try {
			//Test pokemon dataset.
			if (DO_POKEMON_DATASET) {
				pokemonEvaluation = new Evaluation(pokemonTrainingDataSet);
				
				if (TEST_AGAINST_TEST_SET)
					pokemonEvaluation.evaluateModel(pokemonClassifier, pokemonTestDataSet);
				else
					pokemonEvaluation.evaluateModel(pokemonClassifier, pokemonTrainingDataSet);
				
				System.out.print("Pokemon " + classifierName + ": ");
				System.out.println(100.0 * pokemonEvaluation.correct() / pokemonEvaluation.numInstances());
			}
			
			//Test customer satisfaction dataset.
			if (DO_CUSTOMER_SATISFACTION_DATASET) {
				customerSatisfactionEvaluation = new Evaluation(customerSatisfactionTrainingDataSet);
				
				if (TEST_AGAINST_TEST_SET)
					customerSatisfactionEvaluation.evaluateModel(customerSatisfactionClassifier, customerSatisfactionTestDataSet);
				else
					customerSatisfactionEvaluation.evaluateModel(customerSatisfactionClassifier, customerSatisfactionTrainingDataSet);
				
				System.out.print("Customer Satisfaction " + classifierName + ": ");
				System.out.println(100.0 * customerSatisfactionEvaluation.correct() / customerSatisfactionEvaluation.numInstances());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	private static void trainSVM() {
		pokemonSVMClassifier = new SMO();
		customerSatisfactionSVMClassifier = new SMO();
		try {
			if (DO_POKEMON_DATASET)
				pokemonSVMClassifier.buildClassifier(pokemonTrainingDataSet);
			
			if (DO_CUSTOMER_SATISFACTION_DATASET) {
				PolyKernel polyKernel = new PolyKernel();
				polyKernel.setExponent(3);
				customerSatisfactionSVMClassifier.setKernel(polyKernel);
				customerSatisfactionSVMClassifier.buildClassifier(customerSatisfactionTrainingDataSet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void trainBoosting() {
		pokemonBoostingClassifier = new AdaBoostM1();
		customerSatisfactionBoostingClassifier = new AdaBoostM1();
		try {
			if (DO_POKEMON_DATASET) {
				pokemonBoostingClassifier.setNumIterations(BOOSTING_ITERATIONS);
				pokemonBoostingClassifier.setWeightThreshold(95);
				RandomForest pokemonRandomForest = new RandomForest();
				pokemonRandomForest.setNumIterations(5);
				pokemonRandomForest.setMaxDepth(3);
				pokemonBoostingClassifier.setClassifier(pokemonRandomForest);
				pokemonBoostingClassifier.buildClassifier(pokemonTrainingDataSet);	
			}
			
			if (DO_CUSTOMER_SATISFACTION_DATASET) {
				customerSatisfactionBoostingClassifier.setNumIterations(BOOSTING_ITERATIONS);
				customerSatisfactionBoostingClassifier.setWeightThreshold(95);
				RandomForest customerSatisfactionRandomForest = new RandomForest();
				customerSatisfactionRandomForest.setNumIterations(5);
				customerSatisfactionRandomForest.setMaxDepth(3);
				customerSatisfactionBoostingClassifier.setClassifier(customerSatisfactionRandomForest);
				customerSatisfactionBoostingClassifier.buildClassifier(customerSatisfactionTrainingDataSet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void trainKNN() {
		pokemonKNNClassifier = new IBk();
		customerSatisfactionKNNClassifier = new IBk();
		try {
			if (DO_POKEMON_DATASET) {
				pokemonKNNClassifier.setKNN(K);
				pokemonKNNClassifier.buildClassifier(pokemonTrainingDataSet);
			}
			
			if (DO_CUSTOMER_SATISFACTION_DATASET) {
				customerSatisfactionKNNClassifier.setKNN(K);
				customerSatisfactionKNNClassifier.buildClassifier(customerSatisfactionTrainingDataSet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void trainNeuralNetwork() {
		if (DO_POKEMON_DATASET) {
			pokemonNeuralNetworkClassifier = new MultilayerPerceptron();
			pokemonNeuralNetworkClassifier.setLearningRate(0.1);
			pokemonNeuralNetworkClassifier.setMomentum(0.2);
			pokemonNeuralNetworkClassifier.setTrainingTime(NEURAL_NET_TRAINING_TIME);
			pokemonNeuralNetworkClassifier.setHiddenLayers("a");
		}
		
		if (DO_CUSTOMER_SATISFACTION_DATASET) {
			customerSatisfactionNeuralNetworkClassifier = new MultilayerPerceptron();
			customerSatisfactionNeuralNetworkClassifier.setLearningRate(0.1);
			customerSatisfactionNeuralNetworkClassifier.setMomentum(0.2);
			customerSatisfactionNeuralNetworkClassifier.setTrainingTime(NEURAL_NET_TRAINING_TIME);
			customerSatisfactionNeuralNetworkClassifier.setHiddenLayers("a");
		}
		
		try {
			if (DO_POKEMON_DATASET)
				pokemonNeuralNetworkClassifier.buildClassifier(pokemonTrainingDataSet);
			if (DO_CUSTOMER_SATISFACTION_DATASET)
				customerSatisfactionNeuralNetworkClassifier.buildClassifier(customerSatisfactionTrainingDataSet);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void trainDecisionTree() {
		pokemonDecisionTreeClassifier = new J48();
		customerSatisfactionDecisionTreeClassifier = new J48();
		try {
			if (DO_POKEMON_DATASET)
				pokemonDecisionTreeClassifier.buildClassifier(pokemonTrainingDataSet);
			if (DO_CUSTOMER_SATISFACTION_DATASET)
				customerSatisfactionDecisionTreeClassifier.buildClassifier(customerSatisfactionTrainingDataSet);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void loadInstances() {
		try {
			pokemonTrainingDataSet = new Instances(new BufferedReader(
					new FileReader("data" + File.separator + POKEMON_TRAINING_DATA_SET)));
			pokemonTestDataSet = new Instances(new BufferedReader(
					new FileReader("data" + File.separator + POKEMON_TEST_DATA_SET)));
			pokemonTrainingDataSet.setClassIndex(pokemonTrainingDataSet.numAttributes() - 1);
			pokemonTestDataSet.setClassIndex(pokemonTestDataSet.numAttributes() - 1);
			
			customerSatisfactionTrainingDataSet = new Instances(new BufferedReader(
					new FileReader("data" + File.separator + CUSTOMER_SATISFACTION_TRAINING_DATA_SET)));
			customerSatisfactionTestDataSet = new Instances(new BufferedReader(
					new FileReader("data" + File.separator + CUSTOMER_SATISFACTION_TEST_DATA_SET)));
			customerSatisfactionTrainingDataSet.setClassIndex(
					customerSatisfactionTrainingDataSet.numAttributes() - 1);
			customerSatisfactionTestDataSet.setClassIndex(
					customerSatisfactionTestDataSet.numAttributes() - 1);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
