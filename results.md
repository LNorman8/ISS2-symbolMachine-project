# Results

|Dataset    |Benchmark|Current Status|Model Limit| K | weightBase | priorScale | gamma | mixAlpha |
|-----------|---------|--------------|-----------|---|------------|------------|-------|----------|
|Hawaiian   | 1.6456  | 1.4847       | 1.1732    | 8 | 2.80       | 0.350      | 6.0   | 0.00     |
|ElecDemand | 1.3387  | 1.7480       | 1.4935    | 6 | 2.40       | 0.400      | 6.0   | 0.35     |
|Dickens    | 1.7674  | 1.8920       | 2.1583    | 10| 3.60       | 0.600      | 12.0  | 0.00     |
|DIAtemp    | 1.6797  | 1.9327       | 1.7351    | 8 | 3.00       | 0.600      | 12.0  | 0.25     |
|DIAwind    | 1.8762  | 1.9405       | 1.3896    | 8 | 2.80       | 0.350      | 6.0   | 0.00     |
|solarWind  | 0.6692  | 0.8279       | 0.7571    | 6 | 2.40       | 0.200      | 6.0   | 0.35     |
|HoustonRain| 0.1012  | 0.0926       | 0.0789    | 4 | 1.50       | 0.150      | 3.0   | 0.35     |

Model limit is computed by running the model with the test sequence used as the training sequence as well, so it is an oracle upper bound rather than a valid benchmark result.
