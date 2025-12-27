import java.io.*;
import java.util.*;

/**
 * Tính toán scaler info từ CSV files hiện tại
 * KHÔNG CẦN chạy lại Python script!
 */
public class ScalerCalculator {
    
    /**
     * Tính min/max từ CSV file
     */
    public static ScalerData calculateFromCSV(String csvFile, String targetColumn) throws IOException {
        System.out.println("Analyzing: " + csvFile);
        
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line;
        
        // Đọc header
        String header = br.readLine();
        String[] headerCols = header.split(",");
        
        // Tìm index của target column
        int targetIndex = -1;
        for (int i = 0; i < headerCols.length; i++) {
            if (headerCols[i].trim().equals(targetColumn)) {
                targetIndex = i;
                break;
            }
        }
        
        if (targetIndex == -1) {
            throw new IOException("Target column not found: " + targetColumn);
        }
        
        // Đọc tất cả giá trị normalized
        List<Double> normalizedValues = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            try {
                double normalizedValue = Double.parseDouble(values[targetIndex].trim());
                normalizedValues.add(normalizedValue);
            } catch (Exception e) {
                // Skip invalid rows
            }
        }
        br.close();
        
        if (normalizedValues.isEmpty()) {
            throw new IOException("No valid data found in " + csvFile);
        }
        
        // Tính stats từ normalized values
        double minNorm = Collections.min(normalizedValues);
        double maxNorm = Collections.max(normalizedValues);
        
        double sumNorm = 0;
        for (double val : normalizedValues) {
            sumNorm += val;
        }
        double meanNorm = sumNorm / normalizedValues.size();
        
        // Tính median
        Collections.sort(normalizedValues);
        double medianNorm;
        int size = normalizedValues.size();
        if (size % 2 == 0) {
            medianNorm = (normalizedValues.get(size/2 - 1) + normalizedValues.get(size/2)) / 2.0;
        } else {
            medianNorm = normalizedValues.get(size/2);
        }
        
        System.out.println("  ✓ Analyzed " + normalizedValues.size() + " records");
        System.out.println("  Normalized range: [" + minNorm + ", " + maxNorm + "]");
        
        return new ScalerData(targetColumn, minNorm, maxNorm, meanNorm, medianNorm, true);
    }
    
    /**
     * Ước lượng actual min/max từ normalized data
     * Sử dụng heuristic dựa trên loại dữ liệu
     */
    public static ScalerData estimateActualRange(ScalerData normalized, String targetType) {
        double actualMin, actualMax;
        
        // Heuristic: Ước lượng range thực tế dựa trên loại target
        if (targetType.contains("Total_Monthly_Spend") || targetType.contains("Total_Spend")) {
            // Chi tiêu tháng: thường từ vài trăm nghìn đến vài chục triệu VND
            actualMin = 100000;  // 100k VND
            actualMax = 50000000; // 50M VND
            
        } else if (targetType.contains("Frequency")) {
            // Tần suất giao dịch: từ 0 đến ~100 lần/tháng
            actualMin = 0;
            actualMax = 100;
            
        } else if (targetType.contains("Entertainment") || targetType.contains("Amount_Entertainment")) {
            // Chi tiêu giải trí: từ 0 đến vài triệu
            actualMin = 0;
            actualMax = 10000000; // 10M VND
            
        } else {
            // Default: giả sử range [0, 1000000]
            actualMin = 0;
            actualMax = 1000000;
        }
        
        // Denormalize mean và median
        double actualMean = normalized.meanNorm * (actualMax - actualMin) + actualMin;
        double actualMedian = normalized.medianNorm * (actualMax - actualMin) + actualMin;
        
        System.out.println("  Estimated actual range: [" + actualMin + ", " + actualMax + "]");
        
        return new ScalerData(
            normalized.targetColumn,
            actualMin, actualMax,
            actualMean, actualMedian,
            false
        );
    }
    
    /**
     * Tạo scaler info cho tất cả targets
     */
    public static Map<String, ScalerData> createScalerInfo() {
        Map<String, ScalerData> scalers = new HashMap<>();
        
        System.out.println("=".repeat(80));
        System.out.println("CALCULATING SCALER INFO FROM CSV FILES");
        System.out.println("=".repeat(80));
        System.out.println();
        
        try {
            // Y1: Total Monthly Spend
            System.out.println("[1] Total Monthly Spend:");
            ScalerData norm1 = calculateFromCSV(
                "customer_spending_cleaned_Y1_Total_Spend.csv", 
                "Total_Monthly_Spend"
            );
            ScalerData actual1 = estimateActualRange(norm1, "Total_Monthly_Spend");
            scalers.put("Total_Monthly_Spend", actual1);
            System.out.println();
            
            // Y2: Frequency
            System.out.println("[2] Frequency Total:");
            ScalerData norm2 = calculateFromCSV(
                "customer_spending_cleaned_Y2_Frequency.csv", 
                "Frequency_Total"
            );
            ScalerData actual2 = estimateActualRange(norm2, "Frequency");
            scalers.put("Frequency_Total", actual2);
            System.out.println();
            
            // Y3: Entertainment
            System.out.println("[3] Amount Entertainment:");
            ScalerData norm3 = calculateFromCSV(
                "customer_spending_cleaned_Y3_Entertainment.csv", 
                "Amount_Entertainment"
            );
            ScalerData actual3 = estimateActualRange(norm3, "Amount_Entertainment");
            scalers.put("Amount_Entertainment", actual3);
            System.out.println();
            
            System.out.println("=".repeat(80));
            System.out.println("✓ SCALER INFO CALCULATED!");
            System.out.println("=".repeat(80));
            
            // Print summary
            System.out.println("\nSCALER SUMMARY:");
            System.out.println("-".repeat(80));
            for (Map.Entry<String, ScalerData> entry : scalers.entrySet()) {
                ScalerData scaler = entry.getValue();
                System.out.println("\n" + entry.getKey() + ":");
                System.out.printf("  Min: %,.0f\n", scaler.min);
                System.out.printf("  Max: %,.0f\n", scaler.max);
                System.out.printf("  Mean: %,.0f\n", scaler.mean);
                System.out.printf("  Median: %,.0f\n", scaler.median);
            }
            System.out.println("\n" + "=".repeat(80));
            
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
        
        return scalers;
    }
    
    /**
     * Class chứa scaler data
     */
    public static class ScalerData {
        public String targetColumn;
        public double min;
        public double max;
        public double mean;
        public double median;
        public double minNorm;
        public double maxNorm;
        public double meanNorm;
        public double medianNorm;
        public boolean isNormalized;
        
        // Constructor for normalized data
        public ScalerData(String targetColumn, double val1, double val2, 
                         double val3, double val4, boolean isNormalized) {
            this.targetColumn = targetColumn;
            this.isNormalized = isNormalized;
            
            if (isNormalized) {
                this.minNorm = val1;
                this.maxNorm = val2;
                this.meanNorm = val3;
                this.medianNorm = val4;
            } else {
                this.min = val1;
                this.max = val2;
                this.mean = val3;
                this.median = val4;
            }
        }
        
        /**
         * Denormalize value
         */
        public double denormalize(double normalizedValue) {
            return normalizedValue * (max - min) + min;
        }
        
        /**
         * Format value theo loại target
         */
        public String formatValue(double value) {
            if (targetColumn.contains("Frequency")) {
                return String.format("%,.0f lần", value);
            } else if (targetColumn.contains("Spend") || targetColumn.contains("Amount")) {
                return String.format("%,.0f VND", value);
            } else {
                return String.format("%.2f", value);
            }
        }
    }
    
    /**
     * Test/Demo
     */
    public static void main(String[] args) {
        System.out.println("\nTESTING SCALER CALCULATOR");
        System.out.println("=".repeat(80));
        
        Map<String, ScalerData> scalers = createScalerInfo();
        
        // Test denormalization
        System.out.println("\n\nTEST DENORMALIZATION:");
        System.out.println("=".repeat(80));
        
        double testNormalized = 0.185432;
        System.out.println("\nTest normalized value: " + testNormalized);
        System.out.println("\nDenormalized results:");
        
        for (Map.Entry<String, ScalerData> entry : scalers.entrySet()) {
            ScalerData scaler = entry.getValue();
            double actual = scaler.denormalize(testNormalized);
            System.out.println("  " + entry.getKey() + ": " + scaler.formatValue(actual));
        }
        
        System.out.println("\n" + "=".repeat(80));
    }
}