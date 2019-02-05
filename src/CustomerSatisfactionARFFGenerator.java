import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Random;

public class CustomerSatisfactionARFFGenerator {
	private static final int NUM_CUSTOMERS_TEST = 1000;
	
	//Creates two arff files. One contains the training data for the pokemon and the other 
	//contains the test data.
	public static void generateFile(int numCustomersTraining, boolean showOutput) {
		File training = new File("data" + File.separator + Main.CUSTOMER_SATISFACTION_TRAINING_DATA_SET);
		File test = new File("data" + File.separator + Main.CUSTOMER_SATISFACTION_TEST_DATA_SET);
		
		try {
			for (File file : new File[] { training, test } )
			{	
				FileOutputStream outputStream = new FileOutputStream(file);
				BufferedWriter bufferedWriter 
					= new BufferedWriter(new OutputStreamWriter(outputStream));
				
				WriteLine(bufferedWriter, "@RELATION customer_satisfaction");
				WriteLine(bufferedWriter, "");
				
				WriteLine(bufferedWriter, "@ATTRIBUTE	gender	{male,female}");
				WriteLine(bufferedWriter, "@ATTRIBUTE	age		REAL");
				WriteLine(bufferedWriter, "@ATTRIBUTE	income	REAL");
				WriteLine(bufferedWriter, "@ATTRIBUTE	race	{asian,black,latino,white}");
				WriteLine(bufferedWriter, "@ATTRIBUTE	satisfaction_level	{very_unsatisfied,unsatisfied,indifferent,satisfied,very_satisfied}");
				WriteLine(bufferedWriter, "");
				
				Random rand = new Random();
				WriteLine(bufferedWriter, "@DATA");
				
				int numCustomers = numCustomersTraining;
				if (file.equals(test))
					numCustomers = NUM_CUSTOMERS_TEST;
								
				for (int i = 0; i < numCustomers; i++) {
					String gender = "male";
					if (rand.nextBoolean())
						gender = "female";
					
					int age = rand.nextInt(100 - 18) + 18;
					int income = (rand.nextInt(250 - 25) + 25) * 1000;
					int raceInt = rand.nextInt(4);
					String race = "asian";
					if (raceInt == 1)
						race = "black";
					else if (raceInt == 2)
						race = "latino";
					else if (raceInt == 3)
						race = "white";
					
					float satisfactionFromGender = gender.equals("female") ? 1f : .9f;					
					float satisfactionFromAge = 1f - (Math.abs(40 - age) / 150.0f);
					satisfactionFromAge = Math.max(satisfactionFromAge, 0);
					float satisfactionFromIncome = 1f - (Math.abs(100000 - income) / 200000f);
					float totalSatisfaction = satisfactionFromGender * satisfactionFromAge * 
							satisfactionFromIncome;
					totalSatisfaction += rand.nextGaussian() * .05f;
					
					String satisfaction = "very_satisfied";
					if (totalSatisfaction < .2) 
						satisfaction = "very_unsatisfied";
					else if (totalSatisfaction < .4)
						satisfaction = "unsatisfied";
					else if (totalSatisfaction < .6)
						satisfaction = "indifferent";
					else if (totalSatisfaction < .8)
						satisfaction = "satisfied";
					
					WriteCustomer(bufferedWriter, gender, age, income, race, satisfaction);
				}
				
				bufferedWriter.close();
				if (showOutput)
					System.out.println("Finished writing to file " + file.getName());
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void WriteLine(BufferedWriter bufferedWriter, String line) throws IOException {
		bufferedWriter.write(line);
		bufferedWriter.newLine();
	}
	
	private static void WriteCustomer(BufferedWriter bufferedWriter, String gender, int age, 
	int income, String race, String satisfactionLevel) throws IOException {
		WriteLine(bufferedWriter,  gender + "," + age + "," +  income + "," + race + "," +
			satisfactionLevel);
	}
}
