import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Data loader for time series format
 * Multiple rows per account, supports train/validation/test split
 */
public class DataLoader {
    
    /**
     * Load time series dataset
     */
    public static Dataset loadFromCSV(String filename, String targetCol) throws IOException {
        List<Integer> accountKeysList = new ArrayList<>();
        List<Integer> yearsList = new ArrayList<>();
        List<Integer> monthsList = new ArrayList<>();
        List<double[]> featuresList = new ArrayList<>();
        List<Double> targetList = new ArrayList<>();
        
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        
        // Read header
        String header = br.readLine();
        String[] headerCols = header.split(",");
        
        // Build column index map
        Map<String, Integer> colIndexMap = new HashMap<>();
        for (int i = 0; i < headerCols.length; i++) {
            colIndexMap.put(headerCols[i].trim(), i);
        }
        
        System.out.println("Loading: " + filename);
        System.out.println("Target column: " + targetCol);
        
        // Identify feature columns
        List<String> featureColNames = new ArrayList<>();
        for (String col : headerCols) {
            col = col.trim();
            if (!col.equals("Account_Key") && !col.equals("Year") && 
                !col.equals("Month") && !col.equals("Quarter") && 
                !col.equals(targetCol)) {
                featureColNames.add(col);
            }
        }
        
        System.out.println("Number of features: " + featureColNames.size());
        System.out.println();
        
        // Read data
        int lineCount = 0;
        int futureCount = 0;
        
        while ((line = br.readLine()) != null) {
            lineCount++;
            String[] values = line.split(",");
            
            try {
                // Read identifiers
                int accountKey = Integer.parseInt(values[colIndexMap.get("Account_Key")].trim());
                int year = Integer.parseInt(values[colIndexMap.get("Year")].trim());
                int month = Integer.parseInt(values[colIndexMap.get("Month")].trim());
                
                // Read target (may be NaN for future predictions)
                String targetStr = values[colIndexMap.get(targetCol)];
                Double target = null;
                if (!targetStr.isEmpty() && !targetStr.equalsIgnoreCase("nan")) {
                    target = Double.parseDouble(targetStr);
                } else {
                    futureCount++;
                }
                
                // Read features
                double[] features = new double[featureColNames.size()];
                for (int i = 0; i < featureColNames.size(); i++) {
                    String featureCol = featureColNames.get(i);
                    features[i] = Double.parseDouble(values[colIndexMap.get(featureCol)]);
                }
                
                accountKeysList.add(accountKey);
                yearsList.add(year);
                monthsList.add(month);
                featuresList.add(features);
                targetList.add(target);
                
            } catch (Exception e) {
                System.err.println("Error parsing line " + lineCount);
                System.err.println("Error: " + e.getMessage());
            }
        }
        
        br.close();
        
        // Convert to arrays
        int[] accountKeys = accountKeysList.stream().mapToInt(Integer::intValue).toArray();
        int[] years = yearsList.stream().mapToInt(Integer::intValue).toArray();
        int[] months = monthsList.stream().mapToInt(Integer::intValue).toArray();
        double[][] X = featuresList.toArray(new double[0][]);
        Double[] yBoxed = targetList.toArray(new Double[0]);
        
        System.out.println("Loaded " + X.length + " rows with " + X[0].length + " features");
        System.out.println("Training rows (target != null): " + (X.length - futureCount));
        System.out.println("Future rows (target = null): " + futureCount);
        System.out.println();
        
        return new Dataset(accountKeys, years, months, X, yBoxed, targetCol);
    }
    
    /**
     * Split data into train/validation/test sets
     * Only uses rows with actual target values (not future predictions)
     */
    public static Dataset[] splitTrainValTest(Dataset dataset, 
                                              double trainRatio, 
                                              double valRatio) {
        // Filter out future rows (target = null)
        List<Integer> trainingIndices = new ArrayList<>();
        for (int i = 0; i < dataset.y.length; i++) {
            if (dataset.y[i] != null) {
                trainingIndices.add(i);
            }
        }
        
        int totalTraining = trainingIndices.size();
        int trainSize = (int) (totalTraining * trainRatio);
        int valSize = (int) (totalTraining * valRatio);
        int testSize = totalTraining - trainSize - valSize;
        
        System.out.println("Splitting training data:");
        System.out.println("  Train: " + trainSize + " (" + (trainRatio * 100) + "%)");
        System.out.println("  Validation: " + valSize + " (" + (valRatio * 100) + "%)");
        System.out.println("  Test: " + testSize + " (" + ((1 - trainRatio - valRatio) * 100) + "%)");
        
        // Create train set
        int[] accountKeys_train = new int[trainSize];
        int[] years_train = new int[trainSize];
        int[] months_train = new int[trainSize];
        double[][] X_train = new double[trainSize][];
        double[] y_train = new double[trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = trainingIndices.get(i);
            accountKeys_train[i] = dataset.accountKeys[idx];
            years_train[i] = dataset.years[idx];
            months_train[i] = dataset.months[idx];
            X_train[i] = dataset.X[idx];
            y_train[i] = dataset.y[idx];
        }
        
        // Create validation set
        int[] accountKeys_val = new int[valSize];
        int[] years_val = new int[valSize];
        int[] months_val = new int[valSize];
        double[][] X_val = new double[valSize][];
        double[] y_val = new double[valSize];
        
        for (int i = 0; i < valSize; i++) {
            int idx = trainingIndices.get(trainSize + i);
            accountKeys_val[i] = dataset.accountKeys[idx];
            years_val[i] = dataset.years[idx];
            months_val[i] = dataset.months[idx];
            X_val[i] = dataset.X[idx];
            y_val[i] = dataset.y[idx];
        }
        
        // Create test set
        int[] accountKeys_test = new int[testSize];
        int[] years_test = new int[testSize];
        int[] months_test = new int[testSize];
        double[][] X_test = new double[testSize][];
        double[] y_test = new double[testSize];
        
        for (int i = 0; i < testSize; i++) {
            int idx = trainingIndices.get(trainSize + valSize + i);
            accountKeys_test[i] = dataset.accountKeys[idx];
            years_test[i] = dataset.years[idx];
            months_test[i] = dataset.months[idx];
            X_test[i] = dataset.X[idx];
            y_test[i] = dataset.y[idx];
        }
        
        // Convert back to Double[] for compatibility
        Double[] y_train_boxed = new Double[trainSize];
        Double[] y_val_boxed = new Double[valSize];
        Double[] y_test_boxed = new Double[testSize];
        
        for (int i = 0; i < trainSize; i++) y_train_boxed[i] = y_train[i];
        for (int i = 0; i < valSize; i++) y_val_boxed[i] = y_val[i];
        for (int i = 0; i < testSize; i++) y_test_boxed[i] = y_test[i];
        
        return new Dataset[] {
            new Dataset(accountKeys_train, years_train, months_train, X_train, y_train_boxed, dataset.targetCol),
            new Dataset(accountKeys_val, years_val, months_val, X_val, y_val_boxed, dataset.targetCol),
            new Dataset(accountKeys_test, years_test, months_test, X_test, y_test_boxed, dataset.targetCol)
        };
    }
    
    /**
     * Get all future rows (target = null) for prediction
     */
    public static Dataset getFutureData(Dataset dataset) {
        List<Integer> futureIndices = new ArrayList<>();
        for (int i = 0; i < dataset.y.length; i++) {
            if (dataset.y[i] == null) {
                futureIndices.add(i);
            }
        }
        
        int futureSize = futureIndices.size();
        
        int[] accountKeys_future = new int[futureSize];
        int[] years_future = new int[futureSize];
        int[] months_future = new int[futureSize];
        double[][] X_future = new double[futureSize][];
        Double[] y_future = new Double[futureSize];
        
        for (int i = 0; i < futureSize; i++) {
            int idx = futureIndices.get(i);
            accountKeys_future[i] = dataset.accountKeys[idx];
            years_future[i] = dataset.years[idx];
            months_future[i] = dataset.months[idx];
            X_future[i] = dataset.X[idx];
            y_future[i] = null;
        }
        
        return new Dataset(accountKeys_future, years_future, months_future, 
                          X_future, y_future, dataset.targetCol);
    }
    
    /**
     * Dataset class
     */
    public static class Dataset {
        public int[] accountKeys;
        public int[] years;
        public int[] months;
        public double[][] X;
        public Double[] y;  // Using Double to allow null for future predictions
        public String targetCol;
        
        public Dataset(int[] accountKeys, int[] years, int[] months,
                      double[][] X, Double[] y, String targetCol) {
            this.accountKeys = accountKeys;
            this.years = years;
            this.months = months;
            this.X = X;
            this.y = y;
            this.targetCol = targetCol;
        }
        
        /**
         * Convert y to primitive double[] (only non-null values)
         */
        public double[] getYPrimitive() {
            int count = 0;
            for (Double val : y) {
                if (val != null) count++;
            }
            
            double[] result = new double[count];
            int idx = 0;
            for (Double val : y) {
                if (val != null) {
                    result[idx++] = val;
                }
            }
            return result;
        }
    }
}