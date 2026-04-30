# Results

|Dataset    |Benchmark|Current Status|Model Limit | K | weightBase | priorScale | gamma | mixAlpha |
|-----------|---------|--------------|------------|---|------------|------------|-------|----------|
|Hawaiian   | 1.6456  | 1.4847       | 1.1732     | 8 | 2.80       | 0.350      | 6.0   | 0.00     |
|ElecDemand | 1.3387  | 1.8821       | 1.4895     | 6 | 2.40       | 0.600      | 6.0   | 0.00     |
|Dickens    | 1.7674  | 1.8920       | **2.1583** | 10| 3.60       | 0.600      | 12.0  | 0.00     |
|DIAtemp    | 1.6797  | 1.9937       | 1.7354     | 8 | 2.40       | 0.600      | 8.0   | 0.00     |
|DIAwind    | 1.8762  | 1.9405       | 1.3889     | 8 | 2.80       | 0.350      | 6.0   | 0.00     |
|solarWind  | 0.6692  | 0.9101       | 0.7685     | 6 | 2.40       | 0.200      | 6.0   | 0.00     |
|HoustonRain| 0.1012  | 0.0929       | 0.0789     | 4 | 1.50       | 0.150      | 3.0   | 0.00     |

Model limit is computed by running the model with the test sequence used as the training sequence as well, so it is an oracle upper bound rather than a valid benchmark result.
Despite that... Dickens' model limit is worse then our full run. I do not understand how that is possible, but it is what it is. 

Originally, the point of the model limit was to say "Our architecture simply will not be able to beat the benchmark on X dataset". However, that's not a good limit.