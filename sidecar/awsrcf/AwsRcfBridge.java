import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ForecastDescriptor;
import com.amazon.randomcutforest.parkservices.RCFCaster;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.Calibration;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * Line protocol bridge between the Python adapter and the AWS Random Cut Forest library.
 * One process hosts any number of series, each with its own forest.
 *
 * <pre>
 * NEW  sid mode key=value ...   -> OK | ERR message          mode: rcf | trcf | caster
 * U    sid timestamp value      -> R score grade threshold expected lower upper ready
 * DEL  sid                      -> OK
 * MEM                           -> M usedBytes (after a full GC)
 * QUIT
 * </pre>
 *
 * Numbers are plain decimal text, NaN is written as "nan".
 */
public final class AwsRcfBridge {

    interface Series {
        String step(long timestamp, double value);
    }

    private static String num(double x) {
        return Double.isNaN(x) || Double.isInfinite(x) ? "nan" : Double.toString(x);
    }

    /** The plain forest of the core library: score before update, no threshold. */
    static final class Plain implements Series {
        private final RandomCutForest forest;

        Plain(Map<String, String> p) {
            int shingle = Integer.parseInt(p.getOrDefault("shingle", "1"));
            RandomCutForest.Builder<?> b = RandomCutForest.builder().dimensions(shingle)
                    .shingleSize(shingle).internalShinglingEnabled(true)
                    .numberOfTrees(Integer.parseInt(p.getOrDefault("trees", "100")))
                    .sampleSize(Integer.parseInt(p.getOrDefault("sample", "256")))
                    .randomSeed(Long.parseLong(p.getOrDefault("seed", "0")))
                    .parallelExecutionEnabled(false);
            if (p.containsKey("decay")) {
                b.timeDecay(Double.parseDouble(p.get("decay")));
            }
            if (p.containsKey("output_after")) {
                b.outputAfter(Integer.parseInt(p.get("output_after")));
            }
            this.forest = b.build();
        }

        @Override
        public String step(long ts, double v) {
            double[] point = {v};
            boolean ready = forest.isOutputReady();
            double score = 0.0;
            if (ready) {
                score = forest.getAnomalyScore(forest.transformToShingledPoint(
                        new float[] {(float) v}));
            }
            forest.update(point);
            return "R " + num(score) + " 0 nan nan nan nan " + (ready ? 1 : 0);
        }
    }

    /** ThresholdedRandomCutForest, and RCFCaster when the forecast is switched on. */
    static final class Thresholded implements Series {
        private final ThresholdedRandomCutForest trcf;
        private final RCFCaster caster;
        private double nextExpected = Double.NaN;
        private double nextLower = Double.NaN;
        private double nextUpper = Double.NaN;

        Thresholded(Map<String, String> p, boolean cast) {
            int shingle = Integer.parseInt(p.getOrDefault("shingle", "8"));
            int sample = Integer.parseInt(p.getOrDefault("sample", "256"));
            int trees = Integer.parseInt(p.getOrDefault("trees", "50"));
            long seed = Long.parseLong(p.getOrDefault("seed", "0"));
            int outputAfter = Integer.parseInt(p.getOrDefault("output_after",
                    Integer.toString(Math.max(32, sample / 4))));
            double rate = Double.parseDouble(p.getOrDefault("anomaly_rate", "0.005"));
            double zFactor = Double.parseDouble(p.getOrDefault("z_factor", "2.5"));
            if (cast) {
                RCFCaster.Builder b = RCFCaster.builder().dimensions(shingle).shingleSize(shingle)
                        .internalShinglingEnabled(true).numberOfTrees(trees).sampleSize(sample)
                        .randomSeed(seed).outputAfter(outputAfter).anomalyRate(rate)
                        .zFactor(zFactor).parallelExecutionEnabled(false)
                        .transformMethod(TransformMethod.NORMALIZE)
                        .forecastHorizon(Integer.parseInt(p.getOrDefault("horizon", "1")))
                        .calibration(Calibration.valueOf(
                                p.getOrDefault("calibration", "SIMPLE")))
                        .percentile(Double.parseDouble(p.getOrDefault("percentile", "0.1")));
                if (p.containsKey("decay")) {
                    b.timeDecay(Double.parseDouble(p.get("decay")));
                }
                if (p.containsKey("error_horizon")) {
                    b.errorHorizon(Integer.parseInt(p.get("error_horizon")));
                }
                this.caster = b.build();
                this.trcf = this.caster;
            } else {
                ThresholdedRandomCutForest.Builder<?> b = ThresholdedRandomCutForest.builder()
                        .dimensions(shingle).shingleSize(shingle).internalShinglingEnabled(true)
                        .numberOfTrees(trees).sampleSize(sample).randomSeed(seed)
                        .outputAfter(outputAfter).anomalyRate(rate).zFactor(zFactor)
                        .parallelExecutionEnabled(false)
                        .transformMethod(TransformMethod.NORMALIZE);
                if (p.containsKey("decay")) {
                    b.timeDecay(Double.parseDouble(p.get("decay")));
                }
                this.trcf = b.build();
                this.caster = null;
            }
        }

        @Override
        public String step(long ts, double v) {
            // The band for this sample is the forecast made after the previous one.
            double expected = nextExpected;
            double lower = nextLower;
            double upper = nextUpper;
            AnomalyDescriptor d = trcf.process(new double[] {v}, ts);
            if (caster != null && d instanceof ForecastDescriptor f
                    && f.getTimedForecast() != null
                    && f.getTimedForecast().rangeVector != null
                    && f.getTimedForecast().rangeVector.values.length > 0
                    && f.getTimedForecast().rangeVector.lower[0] != f.getTimedForecast().rangeVector.upper[0]) {
                nextExpected = f.getTimedForecast().rangeVector.values[0];
                nextLower = f.getTimedForecast().rangeVector.lower[0];
                nextUpper = f.getTimedForecast().rangeVector.upper[0];
            }
            boolean ready = d.getRCFScore() > 0 || d.getThreshold() > 0;
            return "R " + num(d.getRCFScore()) + " " + num(d.getAnomalyGrade()) + " "
                    + num(d.getThreshold()) + " " + num(expected) + " " + num(lower) + " "
                    + num(upper) + " " + (ready ? 1 : 0);
        }
    }

    private static Map<String, String> params(String[] parts, int from) {
        Map<String, String> m = new HashMap<>();
        for (int i = from; i < parts.length; i++) {
            int eq = parts[i].indexOf('=');
            if (eq > 0) {
                m.put(parts[i].substring(0, eq), parts[i].substring(eq + 1));
            }
        }
        return m;
    }

    public static void main(String[] args) throws Exception {
        BufferedReader in = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8), 1 << 16);
        PrintStream out = new PrintStream(new java.io.BufferedOutputStream(System.out, 1 << 16),
                false, StandardCharsets.UTF_8);
        Map<String, Series> all = new HashMap<>();
        out.println("READY");
        out.flush();
        String line;
        while ((line = in.readLine()) != null) {
            String[] parts = line.trim().split("\\s+");
            String reply;
            try {
                switch (parts[0]) {
                case "NEW": {
                    Map<String, String> p = params(parts, 3);
                    switch (parts[2]) {
                    case "rcf" -> all.put(parts[1], new Plain(p));
                    case "trcf" -> all.put(parts[1], new Thresholded(p, false));
                    case "caster" -> all.put(parts[1], new Thresholded(p, true));
                    default -> throw new IllegalArgumentException("unknown mode " + parts[2]);
                    }
                    reply = "OK";
                    break;
                }
                case "U": {
                    Series s = all.get(parts[1]);
                    if (s == null) {
                        throw new IllegalArgumentException("unknown series " + parts[1]);
                    }
                    reply = s.step((long) Double.parseDouble(parts[2]),
                            Double.parseDouble(parts[3]));
                    break;
                }
                case "DEL":
                    all.remove(parts[1]);
                    reply = "OK";
                    break;
                case "MEM": {
                    Runtime r = Runtime.getRuntime();
                    System.gc(); // so that the figure is live data, not garbage
                    reply = "M " + (r.totalMemory() - r.freeMemory());
                    break;
                }
                case "QUIT":
                    out.flush();
                    return;
                default:
                    reply = "ERR unknown command";
                }
            } catch (Exception | StackOverflowError e) {
                reply = "ERR " + e.getClass().getSimpleName() + " " + e.getMessage();
            }
            out.println(reply);
            if (!in.ready()) {
                out.flush();
            }
        }
        out.flush();
    }
}
