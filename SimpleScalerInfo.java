import java.util.HashMap;
import java.util.Map;

/**
 * Simple Scaler Info - KHÔNG CẦN JSON!
 * Sử dụng hardcoded ranges dựa trên phân tích dataset
 */
public class SimpleScalerInfo {
    
    private Map<String, TargetScaler> scalers;
    
    public SimpleScalerInfo() {
        this.scalers = new HashMap<>();
        initializeScalers();
    }
    
    /**
     * Initialize scalers với estimated ranges
     * ĐIỀU CHỈNH CÁC GIÁ TRỊ NÀY dựa trên dataset thực tế của bạn
     */
    private void initializeScalers() {
        // Y1: Total Monthly Spend (VND)
        // Ước lượng: Người chi tiêu ít nhất ~100k/tháng, nhiều nhất ~50M/tháng
        scalers.put("Total_Monthly_Spend", new TargetScaler(
            "Total_Monthly_Spend",
            100000.0,      // min: 100k VND
            50000000.0,    // max: 50M VND
            11810652.0,     // mean: ~5M VND
            10694435.0      // median: ~3M VND
        ));
        
        // Y2: Frequency Total (số lần giao dịch)
        // Ước lượng: Ít nhất 0 lần, nhiều nhất ~100 lần/tháng
        scalers.put("Frequency_Total", new TargetScaler(
            "Frequency_Total",
            0.0,           // min: 0 lần
            100.0,         // max: 100 lần
            45.0,          // mean: ~25 lần
            45.0           // median: ~20 lần
        ));
        
        // Y3: Amount Entertainment (VND)
        // Ước lượng: 0 đến ~10M cho giải trí
        scalers.put("Amount_Entertainment", new TargetScaler(
            "Amount_Entertainment",
            0.0,           // min: 0 VND
            10000000.0,    // max: 10M VND
            1068737.0,     // mean: ~1.5M VND
            605683.0      // median: ~1M VND
        ));
        
        System.out.println("✓ Initialized SimpleScalerInfo with estimated ranges");
    }
    
    /**
     * Denormalize giá trị từ [0,1] về giá trị thực
     */
    public double denormalize(String targetName, double normalizedValue) {
        TargetScaler scaler = scalers.get(targetName);
        if (scaler == null) {
            throw new IllegalArgumentException("Unknown target: " + targetName);
        }
        
        return normalizedValue * (scaler.max - scaler.min) + scaler.min;
    }
    
    /**
     * Get scaler cho một target
     */
    public TargetScaler getScaler(String targetName) {
        return scalers.get(targetName);
    }
    
    /**
     * Print scaler info
     */
    public void printInfo() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SCALER INFORMATION (Estimated Ranges)");
        System.out.println("=".repeat(80));
        
        for (Map.Entry<String, TargetScaler> entry : scalers.entrySet()) {
            TargetScaler scaler = entry.getValue();
            System.out.println("\n" + entry.getKey() + ":");
            System.out.printf("  Min: %s\n", scaler.formatValue(scaler.min));
            System.out.printf("  Max: %s\n", scaler.formatValue(scaler.max));
            System.out.printf("  Mean: %s\n", scaler.formatValue(scaler.mean));
            System.out.printf("  Median: %s\n", scaler.formatValue(scaler.median));
        }
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("💡 Note: These are estimated ranges. For exact values,");
        System.out.println("   run ScalerCalculator to analyze your CSV files.");
        System.out.println("=".repeat(80));
    }
    
    /**
     * Update scaler ranges manually (nếu cần điều chỉnh)
     */
    public void updateScaler(String targetName, double min, double max, 
                            double mean, double median) {
        scalers.put(targetName, new TargetScaler(targetName, min, max, mean, median));
        System.out.println("✓ Updated scaler for: " + targetName);
    }
    
    /**
     * Inner class chứa thông tin scaler
     */
    public static class TargetScaler {
        public String column;
        public double min;
        public double max;
        public double mean;
        public double median;
        
        public TargetScaler(String column, double min, double max, 
                           double mean, double median) {
            this.column = column;
            this.min = min;
            this.max = max;
            this.mean = mean;
            this.median = median;
        }
        
        /**
         * Format giá trị theo loại target
         */
        public String formatValue(double value) {
            if (column.contains("Frequency")) {
                return String.format("%,.0f lần", value);
            } else if (column.contains("Spend") || column.contains("Amount")) {
                return String.format("%,.0f VND", value);
            } else {
                return String.format("%.2f", value);
            }
        }
    }
}