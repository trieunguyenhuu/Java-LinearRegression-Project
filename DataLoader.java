import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Data loader for focused time series data
 * Each target Y has its own dataset with only relevant lag features
 */
public class DataLoader {
    
    /**
     * Load focused data for a specific target variable
     * @param filename Path to cleaned CSV file (e.g., customer_spending_cleaned_Y1_Total_Spend.csv)
     * @return Dataset object
     */
    public static Dataset loadFromCSV(String filename, String targetCol) throws IOException {
        List<Integer> accountKeysList = new ArrayList<>();
        List<Integer> yearsList = new ArrayList<>();
        List<Integer> monthsList = new ArrayList<>();
        List<double[]> featuresList = new ArrayList<>();
        List<Double> targetList = new ArrayList<>();
        
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        
        // Đọc header
        String header = br.readLine();
        String[] headerCols = header.split(",");
        
        // Build column index map
        Map<String, Integer> colIndexMap = new HashMap<>();
        for (int i = 0; i < headerCols.length; i++) {
            colIndexMap.put(headerCols[i].trim(), i);
        }
        
        System.out.println("Loading: " + filename);
        System.out.println("Target column: " + targetCol);
        
        // Identify feature columns (exclude identifiers and target)
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
        System.out.println("Features: " + featureColNames);
        System.out.println();
        
        // Đọc dữ liệu
        int lineCount = 0;
        while ((line = br.readLine()) != null) {
            lineCount++;
            String[] values = line.split(",");
            
            try {
                // Read identifiers
                int accountKey = Integer.parseInt(values[colIndexMap.get("Account_Key")].trim());
                int year = Integer.parseInt(values[colIndexMap.get("Year")].trim());
                int month = Integer.parseInt(values[colIndexMap.get("Month")].trim());
                
                // Read target
                double target = Double.parseDouble(values[colIndexMap.get(targetCol)]);
                
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
        double[] y = targetList.stream().mapToDouble(Double::doubleValue).toArray();
        
        System.out.println("Loaded " + X.length + " samples with " + X[0].length + " features");
        System.out.println();
        
        return new Dataset(accountKeys, years, months, X, y, targetCol);
    }
    
    /**
     * Split data into train and test sets
     */
    public static Dataset[] splitTrainTest(Dataset dataset, double trainRatio) {
        int totalSamples = dataset.X.length;
        int trainSize = (int) (totalSamples * trainRatio);
        int testSize = totalSamples - trainSize;
        
        // Train set
        int[] accountKeys_train = new int[trainSize];
        int[] years_train = new int[trainSize];
        int[] months_train = new int[trainSize];
        double[][] X_train = new double[trainSize][];
        double[] y_train = new double[trainSize];
        
        // Test set
        int[] accountKeys_test = new int[testSize];
        int[] years_test = new int[testSize];
        int[] months_test = new int[testSize];
        double[][] X_test = new double[testSize][];
        double[] y_test = new double[testSize];
        
        for (int i = 0; i < trainSize; i++) {
            accountKeys_train[i] = dataset.accountKeys[i];
            years_train[i] = dataset.years[i];
            months_train[i] = dataset.months[i];
            X_train[i] = dataset.X[i];
            y_train[i] = dataset.y[i];
        }
        
        for (int i = 0; i < testSize; i++) {
            accountKeys_test[i] = dataset.accountKeys[trainSize + i];
            years_test[i] = dataset.years[trainSize + i];
            months_test[i] = dataset.months[trainSize + i];
            X_test[i] = dataset.X[trainSize + i];
            y_test[i] = dataset.y[trainSize + i];
        }
        
        return new Dataset[] {
            new Dataset(accountKeys_train, years_train, months_train, X_train, y_train, dataset.targetCol),
            new Dataset(accountKeys_test, years_test, months_test, X_test, y_test, dataset.targetCol)
        };
    }
    
    /**
     * Dataset class for focused features
     */
    public static class Dataset {
        public int[] accountKeys;
        public int[] years;
        public int[] months;
        public double[][] X;  // Features (already normalized)
        public double[] y;    // Target (already normalized)
        public String targetCol;
        
        // Map (Account_Key, Year, Month) to index
        public Map<String, Integer> recordKeyToIndex;
        
        public Dataset(int[] accountKeys, int[] years, int[] months, 
                      double[][] X, double[] y, String targetCol) {
            this.accountKeys = accountKeys;
            this.years = years;
            this.months = months;
            this.X = X;
            this.y = y;
            this.targetCol = targetCol;
            
            // Build map
            this.recordKeyToIndex = new HashMap<>();
            for (int i = 0; i < accountKeys.length; i++) {
                String key = accountKeys[i] + "_" + years[i] + "_" + months[i];
                recordKeyToIndex.put(key, i);
            }
        }
        
        /**
         * Get features by Account_Key, Year, Month
         */
        public double[] getFeaturesByKey(int accountKey, int year, int month) {
            String key = accountKey + "_" + year + "_" + month;
            Integer index = recordKeyToIndex.get(key);
            return index != null ? X[index] : null;
        }
        
        /**
         * Get actual value by Account_Key, Year, Month
         */
        public Double getActualByKey(int accountKey, int year, int month) {
            String key = accountKey + "_" + year + "_" + month;
            Integer index = recordKeyToIndex.get(key);
            return index != null ? y[index] : null;
        }
        
        /**
         * Get the latest record for an account
         */
        public Integer getLatestIndexByAccount(int accountKey) {
            Integer latestIndex = null;
            int latestYear = 0;
            int latestMonth = 0;
            
            for (int i = 0; i < accountKeys.length; i++) {
                if (accountKeys[i] == accountKey) {
                    if (years[i] > latestYear || 
                        (years[i] == latestYear && months[i] > latestMonth)) {
                        latestYear = years[i];
                        latestMonth = months[i];
                        latestIndex = i;
                    }
                }
            }
            
            return latestIndex;
        }
    }
}