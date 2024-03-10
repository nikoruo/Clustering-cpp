#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <random>
#include  <iterator>
#include <chrono>

//Constants for file locations,
const std::string DATA_FOLDER = "data/";
const std::string OUTPUTS_FOLDER = "outputs/";
const std::string DATA_FILENAME = DATA_FOLDER + "s1.txt";
const std::string CENTROID_FILENAME = OUTPUTS_FOLDER + "centroid.txt";
const std::string PARTITION_FILENAME = OUTPUTS_FOLDER + "partition.txt";
const char SEPARATOR = ' ';

//and for clustering
const int NUM_CENTROIDS = 15;  // klustereiden lukumäärä: s4 = 15, unbalanced = 8
const int MAX_ITERATIONS = 100; // k-means rajoitus
const int MAX_REPEATS = 25; // repeated kmeans
const int MAX_SWAPS = 100; // 1 kmeans on noin 13 swapsia


class DataPoint {
public:
    std::vector<int> attributes;

    // Minimum distance from this data point to a centroid
	// ei käytössä
    double minDistance = std::numeric_limits<double>::max();

	//For subsets
	bool obsolete = false;
	int ogIndex = -1;

    // Default constructor
	DataPoint() {};
    // Constructor to initialize the data point with given attributes
	explicit DataPoint(const std::vector<int>& attributes) : attributes(attributes) {};
	
	// Overload equality operator for attribute comparison
    bool operator==(const DataPoint& other) const {
        return attributes == other.attributes;
    }

	// Overload inequality operator for attribute comparison
	bool operator!=(const DataPoint& other) const {
		return attributes != other.attributes;
	}
};

// Function to select a random index
template<typename Iter>
size_t select_randomly(Iter start, Iter end) {
	static std::random_device rd; // Initialize random device
	static std::mt19937 gen(rd()); // Initialize Mersenne Twister pseudo-random number generator
	std::uniform_int_distribution<size_t> dis(0, std::distance(start, end) - 1); // Initialize uniform distribution for selecting random index

	return dis(gen);
}

double calculateSquaredEuclideanDistance(const DataPoint& point1, const DataPoint& point2) {
	double sum = 0.0;
	for (size_t i = 0; i < point1.attributes.size(); ++i) {
		sum += std::pow(point1.attributes[i] - point2.attributes[i], 2);
	}
	return sqrt(sum);
}

double calculateEuclideanDistance(const DataPoint& point1, const DataPoint& point2) {
	double sum = 0.0;
	for (size_t i = 0; i < point1.attributes.size(); ++i) {
		sum += std::pow(point1.attributes[i] - point2.attributes[i], 2);
	}
	return sum;
}

// Function for general file error handling
void handleFileError(const std::string& filename) {
	throw std::runtime_error("Error: Unable to open file '" + filename + "'");
}

// Function to read data points from a file
std::vector<DataPoint> readDataPoints(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		handleFileError(filename);
	}

	std::vector<DataPoint> dataPoints;

	// a string to store each line of the file
	std::string line;

	// Read each line of the file
	while (std::getline(file, line)) {
		// Create a string stream to parse the line
		std::istringstream iss(line);
		// Create a vector to store the attributes of the data point
		std::vector<int> attributes;

		// Parse each value from the line and store it in the attributes vector
		int value;
		while (iss >> value) {
			attributes.push_back(value);
		}

		// Add the attributes to the data points vector
		dataPoints.emplace_back(attributes);
	}

	file.close();

	return dataPoints;
}

// Function to get the number of dimensions in the data
int getNumDimensions(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		handleFileError(filename);
	}

	// Read the first line of the file
	std::string line;
	std::getline(file, line);

	// Check if the line is empty
	if (line.empty()) {
		std::cerr << "Error: File '" << filename << "' is empty\n";
		std::exit(EXIT_FAILURE);
	}

	// Count the number of integers in the line
	std::istringstream iss(line);
	int dimensions = std::distance(std::istream_iterator<int>(iss), std::istream_iterator<int>());

	file.close();

	return dimensions;
}

// Function to chooses random data points to be centroids
std::vector<DataPoint> generateRandomCentroids(int numCentroids, const std::vector<DataPoint>& dataPoints) {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	std::vector<DataPoint> dataPointsShuffled(dataPoints.begin(), dataPoints.end());
	std::random_shuffle(dataPointsShuffled.begin(), dataPointsShuffled.end());
	
	std::vector<DataPoint> centroids(numCentroids);

	for (int i = 0; i < numCentroids; ++i) {
		centroids[i] = dataPointsShuffled[i];
	}
	return centroids;
}

// Function to write the centroids to a file
void writeCentroidsToFile(const std::string& filename, const std::vector<DataPoint>& centroids) {
	std::ofstream centroidFile(filename);
	if (!centroidFile.is_open()) {
		handleFileError(filename);
	}

	for (const DataPoint& centroid : centroids) {
		for (double attribute : centroid.attributes) {
			centroidFile << attribute << SEPARATOR;
		}
		centroidFile << "\n";
	}

	centroidFile.close();
	std::cout << "Centroid file created successfully: " << filename << std::endl;
}

//Function to generate random partition (alkuviikkojen tehtäviä)
std::vector<int> generateRandomPartitions(int numDataPoints, int numClusters) {
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	std::vector<int> partitions(numDataPoints);
	for (int i = 0; i < numDataPoints; ++i) {
		partitions[i] = std::rand() % numClusters;
	}
	return partitions;
}

// Function to write partition to a text file
void writePartitionToFile(const std::vector<int>& partition, const std::string& fileName) {
	std::ofstream outFile(fileName);
	if (!outFile.is_open()) {
		handleFileError(fileName);
	}

	// Write each data point and its corresponding cluster to the file
	for (size_t i = 0; i < partition.size(); ++i) {
		outFile << "Data Point " << i << ": Cluster " << partition[i] << "\n";
	}

	outFile.close();

	std::cout << "Optimal partition written to OptimalPartition.txt\n";
}

// Function to calculate SSE (Sum of Squared Errors) of a random centroid and ALL datapoints
double calculateRandomSSE(const std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids) {
	
	int randomCentroidIndex = select_randomly(centroids.begin(), centroids.end());
	const DataPoint& chosenCentroid = centroids[randomCentroidIndex];

	double sse = 0.0;

	for (const DataPoint& dataPoint : dataPoints) {
		sse += calculateEuclideanDistance(dataPoint, chosenCentroid);
	}

	return sse;
}

//Function to calculate the sum of squared errors (SSE)
double calculateSSE(const std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids, const std::vector<int>& partition) {
	if (dataPoints.size() != partition.size()) {
		std::cerr << "Error: Data points and partition size mismatch\n";
		std::exit(EXIT_FAILURE);
	}

	double sse = 0.0;

	for (int i = 0; i < partition.size(); ++i) {
		int cIndex = partition[i];

		if (cIndex >= 0 && cIndex < static_cast<int>(centroids.size())) {
			// SSE between the data point and its assigned centroid
			sse += calculateEuclideanDistance(dataPoints[i], centroids[cIndex]);
		}
		else {
			std::cerr << "Error: Invalid centroid index in partition\n";
			std::exit(EXIT_FAILURE);
		}
	}

	return sse;
}

// Function to calculate pairwise distances and report average distance
void calculateAndReportAverageDistance(const std::vector<DataPoint>& dataPoints) {
	double totalDistance = 0.0;
	int pairwiseCount = 0;

	// Iterate through each pair of data points
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		for (size_t j = i + 1; j < dataPoints.size(); ++j) {
			// Calculate distance between data points i and j
			totalDistance += calculateEuclideanDistance(dataPoints[i], dataPoints[j]);
			pairwiseCount++;
		}
	}

	if (pairwiseCount > 0) {
		double averageDistance = totalDistance / pairwiseCount;
		std::cout << "Average pairwise distance: " << averageDistance << std::endl;
	}
	else {
		std::cerr << "Error: No pairwise distances calculated\n";
	}
}

// Function to find the nearest neighbor of a data point within a set of data
// Käytetään vain findNearestNeighborExample funktiossa
// Mikäli tämän ottaa käyttöön muualla, niin minDistancen päivitys vaatii tarkastelua
DataPoint findNearestNeighbor(DataPoint& queryPoint, const std::vector<DataPoint>& targetPoints) {
	if (targetPoints.empty()) {
		throw std::runtime_error("Error: Cannot find nearest neighbor in an empty set of data");
	}

	double minDistance = std::numeric_limits<double>::max();
	DataPoint nearestNeighbor = queryPoint;

	for (const DataPoint& dataPoint : targetPoints) {
		if (queryPoint != dataPoint) {
			double distance = calculateEuclideanDistance(queryPoint, dataPoint);
			if (distance < minDistance || minDistance == -1) {
				minDistance = distance;
				nearestNeighbor = dataPoint;
			}
		}
	}

	return nearestNeighbor;
}

// Function to find the nearest centroid of a data point
DataPoint findNearestCentroid(DataPoint& queryPoint, const std::vector<DataPoint>& clusterPoints) {
	if (clusterPoints.empty()) {
		throw std::runtime_error("Error: Cannot find nearest centroid in an empty set of data");
	}

	int nearestCentroid = -1;
	std::vector<double> distances(clusterPoints.size());

	for (size_t i = 0; i < clusterPoints.size(); ++i) {
		distances[i] = calculateEuclideanDistance(queryPoint, clusterPoints[i]);	
	}

	double minDistance = distances.front();

	for(int i = 0; i < clusterPoints.size(); ++i){
		if (distances[i] <= minDistance) {
			minDistance = distances[i];
			nearestCentroid = i;
		}
	}

	//Tämä ei tällä hetkellä käytössä, voitaisiin hyödyntää Fast K-means
	queryPoint.minDistance = minDistance;

	return clusterPoints[nearestCentroid];
}

// Function for optimal partitioning
std::vector<int> optimalPartition(std::vector<DataPoint>& dataPoints, const std::vector<DataPoint>& centroids) {
	if (dataPoints.empty() || centroids.empty()) {
		std::cerr << "Error: Cannot perform optimal partition with empty data or centroids\n";
		std::exit(EXIT_FAILURE);
	}

	std::vector<int> partition(dataPoints.size(), -1);

	// Iterate through each data point to find its nearest centroid
	for (size_t i = 0; i < dataPoints.size(); ++i) {
		DataPoint& dataPoint = dataPoints[i];
		DataPoint nearestCentroid;
		
		//debug helper
		//std::cout << "kierros: " << i << std::endl;

		nearestCentroid = findNearestCentroid(dataPoint, centroids);

		//tätä iffittelyä ei varmaan tarvita?
		/*if (std::find(centroids.begin(), centroids.end(), dataPoint) != centroids.end()) {

			nearestCentroid = dataPoint;
		}
		else {
			// Find the nearest centroid for the current data point
		}*/

		// Find the index of the nearest centroid
		auto it = std::find(centroids.begin(), centroids.end(), nearestCentroid);
		if (it != centroids.end()) {
			// And then update the partition with the index of the nearest centroid
			int centroidIndex = std::distance(centroids.begin(), it);
			partition[i] = centroidIndex;
		}
	}

	return partition;
}

// HUOM!BUG!
// random swap + esimerkiksi unbalanced data set sylkee tänne välillä tyhjiä vektoreita
// ja koodi kaatuu siten heti ensimmäiseen iffiin. Tämä vaatii gradun tapauksessa tarkastelua
// Calculate the centroid of a set of data points
DataPoint calculateCentroid(const std::vector<DataPoint>& dataPoints) {
	if (dataPoints.size() == 0) {
		throw std::runtime_error("Cannot calculate centroid for an empty set of data points");
	}

	DataPoint centroid;
	size_t numDimensions = dataPoints.front().attributes.size();

	// Loop through each dimension
	for (size_t dim = 0; dim < numDimensions; ++dim) {
		double sum = 0.0;

		// Calculate the sum of the current dimension
		for (const DataPoint& dataPoint : dataPoints) {
			sum += dataPoint.attributes[dim];
		}

		// Calculate the average for the current dimension and add it to centroid
		centroid.attributes.push_back(sum / dataPoints.size());
	}

	return centroid;
}

// Function to write centroids to a file
// ensimmäisten viikkojen juttuja
void writeCentroidsToFile(const std::vector<DataPoint>& centroids, const std::string& fileName) {
	std::ofstream outFile(fileName);
	if (!outFile.is_open()) {
		handleFileError(fileName);
	}

	for (size_t i = 0; i < centroids.size(); ++i) {
		outFile << "Centroid for Cluster " << i + 1 << ":\n";
		for (size_t dim = 0; dim < centroids[i].attributes.size(); ++dim) {
			outFile << centroids[i].attributes[dim] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

	std::cout << "Constructed centroids written to ConstructedCentroids.txt\n";
}

// Function to perform the centroid step in k-means
std::vector<DataPoint> kMeansCentroidStep(const std::vector<DataPoint>& dataPoints, const std::vector<int>& partition, int numClusters) {
	std::vector<DataPoint> newCentroids(numClusters);
	std::vector<std::vector<DataPoint>> clusters(numClusters);

	for (int i = 0; i < numClusters; ++i) {
		clusters[i] = std::vector<DataPoint>();
	}

	for (int i = 0; i < dataPoints.size(); ++i) {
		int clusterLabel = partition[i];
		clusters[clusterLabel].push_back(dataPoints[i]);
	}

	//BUG
	// RS+unbalanced --> tänne tulee clustereita joissa on 0 datapointtia
	// koodi kaatuu calculateCentroid()
	for (int clusterLabel = 0; clusterLabel < numClusters; ++clusterLabel) {
		newCentroids[clusterLabel] = calculateCentroid(clusters[clusterLabel]);
	}

	return newCentroids;
}

//Function to find a nearest neighbor of a randomly chosen data point
//Ensimmäisten viikkojen juttuja
void findNearestNeighborExample(const std::vector<DataPoint>& dataPoints) {
	DataPoint firstDataPoint = dataPoints[select_randomly(dataPoints.begin(), dataPoints.end())];

	DataPoint nearestNeighbor = findNearestNeighbor(firstDataPoint, dataPoints);

	std::cout << "Nearest neighbor of the first data point:" << std::endl;
	std::cout << "Original Data Point: ";
	for (int value : firstDataPoint.attributes) {
		std::cout << value << " ";
	}
	std::cout << "\nNearest Neighbor: ";
	for (int value : nearestNeighbor.attributes) {
		std::cout << value << " ";
	}
	std::cout << "\n";
}

//Function to run the k-means, returns SSE
// main() pitäisi refactoroida siten, että tämän voi poistaa
// tämä oli ensimmäinen versio runKmeans() funktiosta, joka ei palauta partitionia
double runKMeans(std::vector<DataPoint>& dataPoints, int iterations, std::vector<DataPoint>& centroids) {
	double bestSse = std::numeric_limits<double>::max();
	int stopCounter = 0;
	double previousSSE = std::numeric_limits<double>::max();
	std::vector<int> previousPartition(dataPoints.size(), -1);
	std::vector<int> activeClusters;

	for (int iteration = 0; iteration < iterations; ++iteration) {
		std::vector<int> newPartition = optimalPartition(dataPoints, centroids);
		activeClusters.clear();

		centroids = kMeansCentroidStep(dataPoints, newPartition, NUM_CENTROIDS);

		//Activity = if previous kluster is not the same as the new cluster
		for (int i = 0; i < newPartition.size(); ++i) {

			if (newPartition[i] != previousPartition[i]) {
				if(std::find(activeClusters.begin(), activeClusters.end(), newPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(newPartition[i]);
				}
				if (previousPartition[i] != -1 && std::find(activeClusters.begin(), activeClusters.end(), previousPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(previousPartition[i]);
				}
			}
		}

		double sse = calculateSSE(dataPoints, centroids, newPartition);
		std::cout << "(runKmeans)Total SSE after iteration " << iteration + 1 << ": " << sse << "and activity: " << activeClusters.size() << std::endl;

		if (sse < bestSse) {
			bestSse = sse;
		}
		else if (sse == previousSSE) {
			stopCounter++;
		}

		// For now, we use stopCounter
		// Optionally, check for convergence or other stopping criteria
		if (stopCounter == 3) {
			break;
		}

		previousSSE = sse;
		previousPartition = newPartition;
	}

	return bestSse;
}

//Function to run the k-means, returns SSE and partition
std::pair<double, std::vector<int>> runKMeans(std::vector<DataPoint>& dataPoints, int iterations, std::vector<DataPoint>& centroids, bool foo) {
	double bestSse = std::numeric_limits<double>::max();
	int stopCounter = 0;
	double previousSSE = std::numeric_limits<double>::max();
	std::vector<int> previousPartition(dataPoints.size(), -1);
	std::vector<int> activeClusters;

	for (int iteration = 0; iteration < iterations; ++iteration) {
		std::vector<int> newPartition = optimalPartition(dataPoints, centroids);
		activeClusters.clear();

		centroids = kMeansCentroidStep(dataPoints, newPartition, centroids.size());
		//Activity = if previous kluster is not the same as the new cluster
		for (int i = 0; i < newPartition.size(); ++i) {

			if (newPartition[i] != previousPartition[i]) {
				if (std::find(activeClusters.begin(), activeClusters.end(), newPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(newPartition[i]);
				}
				if (previousPartition[i] != -1 && std::find(activeClusters.begin(), activeClusters.end(), previousPartition[i]) == activeClusters.end())
				{
					activeClusters.push_back(previousPartition[i]);
				}
			}
		}

		// Calculate and report sum-of-squared errors
		double sse = calculateSSE(dataPoints, centroids, newPartition);
		std::cout << "(runKMeans)Total SSE after iteration " << iteration + 1 << ": " << sse << "and activity: " << activeClusters.size() << std::endl;

		if (sse < bestSse) {
			bestSse = sse;
		}
		else if (sse == previousSSE) {
			stopCounter++;
		}

		// For now, we use stopCounter
		// Optionally, check for convergence or other stopping criteria		
		if (stopCounter == 3) {
			break;
		}

		previousSSE = sse;
		previousPartition = newPartition;
	}

	return std::make_pair(bestSse, previousPartition);
}

//Function for random swap
// Deterministic = swaps the centroid to the furthest data point from it
double randomSwap(std::vector<DataPoint>& dataPoints, std::vector<DataPoint>& centroids, bool deterministic) {
	double bestSse = std::numeric_limits<double>::max();
	DataPoint oldCentroid;

	for (int i = 0; i < MAX_SWAPS; ++i) {
		int randomDataPoint = -1;
		double best = -1;

		int randomCentroid = select_randomly(centroids.begin(), centroids.end());
		oldCentroid = centroids[randomCentroid];

		if (deterministic) { //deterministic swap
			for (int j = 0; j < dataPoints.size();++j) {
				if (calculateEuclideanDistance(oldCentroid, dataPoints[j]) > best || j == 0) {
					randomDataPoint = j;
				}
			}
		}
		else { //random swap
			randomDataPoint = select_randomly(dataPoints.begin(), dataPoints.end());
		}

		centroids[randomCentroid] = dataPoints[randomDataPoint];

		double sse = runKMeans(dataPoints, 2, centroids);

		//If SSE improves, we keep the change
		//if not, we reverse the swap
		if (sse < bestSse) {
			bestSse = sse;
		}
		else {
			centroids[randomCentroid] = oldCentroid;
		}
	}

	return bestSse;
}

//Function to calculate the Levenshtein's edit distance
static int editDistance(const std::string& word1, const std::string& word2) {
	int m = word1.size();
	int n = word2.size();

	// Matrix
	std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

	// Initialize the matrix
	for (int i = 0; i <= m; i++) {
		dp[i][0] = i;
	}
	for (int j = 0; j <= n; j++) {
		dp[0][j] = j;
	}

	// Fill the matrix
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			if (word1[i - 1] == word2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + std::min({						
					dp[i - 1][j],		//remove
					dp[i][j - 1],		//insert
					dp[i - 1][j - 1]	//replace
					});
			}
		}
	}

	return dp[m][n];
}

// Function to calculate the pairwise distances between strings in two vectors
static std::vector<std::vector<int>> pairwiseDistances(const std::vector<std::string>& vec1, const std::vector<std::string>& vec2) {
	int n1 = vec1.size();
	int n2 = vec2.size();

	// 2D vector
	std::vector<std::vector<int>> distances(n1, std::vector<int>(n2, 0));

	for (int i = 0; i < n1; ++i) {
		for (int j = 0; j < n2; ++j) {
			distances[i][j] = editDistance(vec1[i], vec2[j]);
		}
	}

	return distances;
}

// Function to calculate pairwise distances within a vector
std::vector<std::vector<int>> pairwiseDistances(const std::vector<std::string>& cluster) {
	std::vector<std::vector<int>> distances(cluster.size(), std::vector<int>(cluster.size(), 0));

	for (size_t i = 0; i < cluster.size(); ++i) {
		for (size_t j = i + 1; j < cluster.size(); ++j) {
			int distance = editDistance(cluster[i], cluster[j]);

			distances[i][j] = distance;
		}
	}

	return distances;
}

// Function to find the medoid for a string cluster
std::pair<std::string, int> findMedoid(const std::vector<std::string>& cluster) {
	int minSumDistance = std::numeric_limits<int>::max();
	std::string medoid;

	// Iterate through each string in the cluster
	for (const std::string& str1 : cluster) {
		int sumDistance = 0;

		// Calculate the sum of distances to other strings in the cluster
		for (const std::string& str2 : cluster) {
			sumDistance += editDistance(str1, str2);
		}

		// Update the medoid if the sum of distances is smaller
		if (sumDistance < minSumDistance) {
			minSumDistance = sumDistance;
			medoid = str1;
		}
	}

	return std::make_pair(medoid, minSumDistance);
}

//Function to calculate Euclidian distance between two centroids multiplied by their sizes
double centroidDistance(const DataPoint& centroid1, const int size1, const DataPoint& centroid2, const int size2) {
	double distance = calculateEuclideanDistance(centroid1, centroid2);

	distance *= size1 * size2;

	return distance;
}

//Function to calculate Pairwise distances between two clusters
double calculatePairwiseDistancesOfClusters(const std::vector<DataPoint>& dataPoints, const std::vector<int>& partition, const int c1, const int c2) {
	double sumDistances = 0.0;
	std::vector<DataPoint> cluster1;
	std::vector<DataPoint> cluster2;


	for (size_t i = 0; i < partition.size(); i++) {
		if (partition[i] == c1) cluster1.push_back(dataPoints[i]);
		else if (partition[i] == c2) cluster2.push_back(dataPoints[i]);
	}

	for (const DataPoint& point1 : cluster1) {
		for (const DataPoint& point2 : cluster2) {
			double distance = calculateEuclideanDistance(point1, point2);
			sumDistances += distance;
		}
	}

	return sumDistances;
}

// Function to find kNN
std::vector<DataPoint> findKNN(DataPoint queryPoint, const std::vector<DataPoint>& targetPoints, int k) {
	if (targetPoints.empty()) {
		throw std::runtime_error("Error: Cannot find nearest neighbors in an empty set of data");
	}

	std::vector<std::pair<double, int>> distancesAndIndices;

	for (const DataPoint& dataPoint : targetPoints) {
		double distance = calculateEuclideanDistance(queryPoint, dataPoint);
		
		auto it = std::find(targetPoints.begin(), targetPoints.end(), dataPoint);
		int index = std::distance(targetPoints.begin(), it);

		distancesAndIndices.emplace_back(distance, index);
	}

	std::vector<std::pair<double, int>> kSmallest(3);

	//Ajattelin, että tämä saattaisi tehostaa koodia N log N --> N
	//mutta ajo kestää yhä äärimmäisen kauan
	/*for (int i = 0; i < distancesAndIndices.size(); i++) {
		if (i>k) {
			if (distancesAndIndices[i].first < kSmallest[k-1].first) {
				kSmallest[k-1] = distancesAndIndices[i];
				std::sort(kSmallest.begin(), kSmallest.end());
			}
		}
		else if (i < k) {
			kSmallest.push_back(distancesAndIndices[i]);
		}
		else if (i == k) std::sort(kSmallest.begin(), kSmallest.end());
	}*/

	std::sort(distancesAndIndices.begin(), distancesAndIndices.end());

	std::vector<DataPoint> kNN;
	for (int i = 0; i < k; ++i) {
		kNN.push_back(targetPoints[distancesAndIndices[i].second]);
	}

	return kNN;
}

//Function for mean-shift
void meanShift(std::vector<DataPoint>& dataPoints, int k) {
	for (DataPoint& queryPoint : dataPoints) {
		std::vector<DataPoint> knn = findKNN(queryPoint, dataPoints, k);

		DataPoint centroid = calculateCentroid(knn);

		queryPoint.attributes = centroid.attributes;

		std::cout << "meanShifted" <<std::endl;
	}
}

//random generator
// number from 0 to 1
double randomGenerator() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	return dis(gen);
}

//Helper function for creating the sub set
//makes % of the data points as "obsolete"
void createSubSet(std::vector<DataPoint>& dataPoints, double percentage) {
	for (DataPoint& point : dataPoints) {
		if (randomGenerator() < percentage) {
			point.obsolete = true;
		}
	}
}

//Helper function for sub sets
// clears the "obsolete" status
//ei käytössä tällä hetkellä
void clearObsoletes(std::vector<DataPoint>& dataPoints) {
	for (DataPoint& point : dataPoints) {
		point.obsolete = false;
	}
}

//Helper function for checking stability
bool vectorContains(const std::vector<int>& vec, int target) {
	for (int i : vec) {
		if (i == target) {
			return true;
		}
	}
	return false;
}

//Function to check stability (for sub set exercises)
bool checkStability(std::vector<DataPoint>& subSet, std::vector<int> subPart, std::vector<int> ogPart) {
	std::vector<int> handledClusters;
	bool stab = true;

	for (int i = 0; i < subSet.size(); ++i) {
		if (!vectorContains(handledClusters, subPart[i])) {

			handledClusters.push_back(subPart[i]);

			//Datapoints of the cluster
			std::vector<int> clusterDataPoints;
			for (int j = 0; j < subSet.size(); ++j) {
				if (subPart[j] == subPart[i]) {
					clusterDataPoints.push_back(subSet[j].ogIndex);
				}
			}

			//Indexes of the original datapoints
			std::vector<int> ogClusterDataPointIndexes;
			for (int j = 0; j < ogPart.size(); ++j) {
				if (ogPart[j] == ogPart[subSet[i].ogIndex]) {

					ogClusterDataPointIndexes.push_back(j);
				}
			}

			int happy = 0; //shares the same cluster in both sets
			int unhappy = 0; //does not share the same cluster in both sets

			//Check if all datapoints in the sub set cluster, also belong to the same cluster in the full set
			for (int ogIndex: clusterDataPoints) {
				if (!vectorContains(ogClusterDataPointIndexes, ogIndex)) {
					stab = false;
					unhappy++;
				}
				else happy++;
			}

			std::cout << "Sub cluster: " << subPart[i] << " --> " << "Happy points: " << happy << " & " << "Unhappy points: " << unhappy << std::endl;
		}
	}

	return stab;
}

//Function for split algorithm
double runSplit(std::vector<DataPoint> dataPoints, int size) {
	std::vector<DataPoint> centroids = generateRandomCentroids(1, dataPoints);
	std::vector<int> partition(dataPoints.size(), 0);
	std::vector <double> sses(dataPoints.size()); //ei käytössä, voisi käyttää tentative valintaan
	double sse = 0; //ei käytössä, voisi käyttää tentative valintaan

	while (centroids.size() != size) {
		int c = select_randomly(centroids.begin(), centroids.end());

		std::vector<int> indexes;
		std::vector<DataPoint> dpoints;

		for (int i = 0; i < dataPoints.size(); ++i) {
			if (partition[i] == c) {
				indexes.push_back(i);
				dpoints.push_back(dataPoints[i]);
			}
		}

		int c1 = indexes[select_randomly(indexes.begin(), indexes.end())];
		int c2 = c1;
		centroids[c] = dataPoints[c1];

		while (c2 == c1) {
			c2 = indexes[select_randomly(indexes.begin(), indexes.end())];
		}
		centroids.push_back(dataPoints[c2]);


		std::vector<DataPoint> newCentroids;
		newCentroids.push_back(dataPoints[c1]);
		newCentroids.push_back(dataPoints[c2]);

		std::pair<double, std::vector<int>> result = runKMeans(dpoints, 5, newCentroids, true);

		auto it = std::find(centroids.begin(), centroids.end(), dataPoints[c1]);
		int c1index =-1;
		
		if (it != centroids.end()) {
			c1index = std::distance(centroids.begin(), it);
		}

		it = std::find(centroids.begin(), centroids.end(), dataPoints[c2]);
		int c2index =-1;
		
		if (it != centroids.end()) {
			c2index = std::distance(centroids.begin(), it);
		}

		for (int i = 0; i < result.second.size(); ++i) {
			partition[indexes[i]] = result.second[i] == 0 ? c1index : c2index;
		}

		sse = result.first;
	}

	std::pair<double, std::vector<int>> result = runKMeans(dataPoints, MAX_ITERATIONS, centroids, true);
	std::cout << "Split without global kmeans : " << calculateSSE(dataPoints, centroids, partition) << std::endl;

	return result.first;
}

int main() {
	int numDimensions = getNumDimensions(DATA_FILENAME);

	//string stuff, medoids
	if (false) {
		// Edit distance between 2 strings
		std::cout << "Edit distance between 2 strings" << std::endl;
		std::string str1 = "kissanpennut";
		std::string str2 = "koiranpentu";
		std::cout << "Edit distance between '" << str1 << "' and '" << str2 << "': " << editDistance(str1, str2) << std::endl;
	
		// Pairwise distances between two clusters
		std::cout << "All pairwise distances between two clusters" << std::endl;
		std::vector<std::string> strCluster1 = { "ireadL", "relanE", "rlanZd", "irelLITnd"};
		std::vector<std::string> strCluster2 = { "fiInVlLand", "filanNM", "finPAlaQd", "finlCnUd"};
		
		std::vector<std::vector<int>> dist = pairwiseDistances(strCluster1, strCluster2);
		int tpd = 0;

		for (int i = 0; i < dist.size(); ++i) {
			for (int j = 0; j < dist[i].size(); ++j) {
				tpd += dist[i][j];
				std::cout << "Distance between '" << strCluster1[i] << "' and '" << strCluster2[j] << "': " << dist[i][j] << std::endl;
			}
		}
		std::cout << "Total pairwise distances:  " << tpd << std::endl;

		// Pairwise distances inside a cluster
		std::cout << "Pairwise distances inside a cluster" << std::endl;
		dist.clear();
		tpd = 0;

		dist = pairwiseDistances(strCluster1);

		// Cluster1
		std::cout << "Pairwise distances within the cluster1:" << std::endl;
		for (size_t i = 0; i < dist.size(); ++i) {
			for (size_t j = 1+i; j < dist[i].size(); ++j) {
				tpd += dist[i][j];
				std::cout << "Distance between '" << strCluster1[i] << "' and '" << strCluster1[j] << "': " << dist[i][j] << std::endl;
			}
		}
		std::cout << "Total pairwise distances:  " << tpd << std::endl;

		// Cluster2
		std::cout << "Pairwise distances within the cluster2:" << std::endl;
		for (size_t i = 0; i < dist.size(); ++i) {
			for (size_t j = 1 + i; j < dist[i].size(); ++j) {
				tpd += dist[i][j];
				std::cout << "Distance between '" << strCluster2[i] << "' and '" << strCluster2[j] << "': " << dist[i][j] << std::endl;
			}
		}
		std::cout << "Total pairwise distances:  " << tpd << std::endl;

		// Medoids and costs
		std::cout << "Medoids and costs" << std::endl;

		std::pair<std::string, int> result = findMedoid(strCluster1);
		std::cout << "Medoid of the cluster1: " << result.first << ", cost: " << result.second << std::endl;

		result = findMedoid(strCluster2);
		std::cout << "Medoid of the cluster2: " << result.first << ", cost: " << result.second << std::endl;
	}

	//kmeans, randomswap, deterministic swap, split, etc
	if(false){
		std::cout << "Number of dimensions in the data: " << numDimensions << std::endl;

		std::vector<DataPoint> dataPoints = readDataPoints(DATA_FILENAME);
		std::cout << "Dataset size: " << dataPoints.size() << std::endl;

		//meanShift --> disabled as it would take hours
		/*for (int i = 0; i < 1; ++i) {
			meanShift(dataPoints, 3);
		}*/

		// Generate and write centroids
		// we also initialize ogCentroids here, so we can compare k-means and random swap from the same starting point
		std::vector<DataPoint> centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);
		std::vector<DataPoint> ogCentroids(centroids.begin(), centroids.end());

		// Liittyy ensimmäisten viikkojen tehtäviin
		writeCentroidsToFile(centroids, CENTROID_FILENAME);
		writePartitionToFile(generateRandomPartitions(dataPoints.size(), NUM_CENTROIDS), PARTITION_FILENAME);
		calculateAndReportAverageDistance(dataPoints);
		// Disabled, as nearestNeighbor updates minDistance --> might cause problems
		//findNearestNeighborExample(dataPoints);
		//Disabled, makes the prints more confusing
		// SSE between a random centroid and ALL data points
		//double sse = calculateRandomSSE(dataPoints, centroids);
		//std::cout << "(Random) Sum-of-Squared Errors (SSE): " << sse << std::endl;

		// Start the clock
		auto start = std::chrono::high_resolution_clock::now();

		// Initial partition and SSE
		std::vector<int> initialPartition = optimalPartition(dataPoints, centroids);
		double initialSSE = calculateSSE(dataPoints, centroids, initialPartition);
		std::cout << "Initial Total Sum-of-Squared Errors (SSE): " << initialSSE << std::endl;

		//Related to centroidDistance() function a bit below
		/*std::vector<int> initialSizes(centroids.size(), 0);
		for (size_t i = 0; i < initialPartition.size(); ++i) {
			initialSizes[initialPartition[i]] += 1;
		}*/ 

		//Helper --> can be used to print the size of individual clusters
		/*for (size_t i = 0; i < centroids.size(); ++i) {
			std::cout << "size: " << initialSizes[i] << std::endl;
		}*/

		//Example: Euclidian distance between two centroids multiplied by their sizes
		//double exDist = centroidDistance(centroids[0], initialSizes[0], centroids[1], initialSizes[1]);
		//std::cout << "Euclidian distance between two centroids multiplied by their sizes: " << exDist << std::endl;

		//Example: Pairwise distances of 2 clusters
		//double exPairwiseDistance = calculatePairwiseDistancesOfClusters(dataPoints, initialPartition, 0, 1);
		//std::cout << "Pairwise distance of 2 clusters: " << exPairwiseDistance << std::endl;

		// Write the initial partition to a text file
		//writePartitionToFile(initialPartition, OUTPUTS_FOLDER + "OptimalPartition.txt");

		//run just k-means
		double bestSse1 = runKMeans(dataPoints, MAX_ITERATIONS, centroids);
		
		// Stop the clock
		auto end = std::chrono::high_resolution_clock::now();
		// Calculate the duration
		std::chrono::duration<double> duration = end - start;
		// Output the duration in seconds
		std::cout << "(K-means)Time taken: " << duration.count() << " seconds" << std::endl;

		double bestSse5 = bestSse1;

		//Repeated k-means
		for (int repeat = 0; repeat < MAX_REPEATS; ++repeat) {
			std::cout << "round: " << repeat << std::endl;

			// New centroids
			if(repeat != 0) centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);

			double newSse = runKMeans(dataPoints, MAX_ITERATIONS, centroids);	

			if (newSse < bestSse5) {
				bestSse5 = newSse;
			}
		}
		
		//Write the constructed centroids to a text file (disabled)
		//writeCentroidsToFile(centroids, OUTPUTS_FOLDER + "ConstructedCentroids.txt");

		// Start the clock
		start = std::chrono::high_resolution_clock::now();

		//Random swap
		double bestSse2 = randomSwap(dataPoints, ogCentroids, false);

		// Stop the clock
		end = std::chrono::high_resolution_clock::now();
		// Calculate the duration
		duration = end - start;
		// Output the duration in seconds
		std::cout << "(Random Swap)Time taken: " << duration.count() << " seconds" << std::endl;

		// Start the clock
		start = std::chrono::high_resolution_clock::now();

		double bestSse3 = randomSwap(dataPoints, ogCentroids, true);

		// Stop the clock
		end = std::chrono::high_resolution_clock::now();
		// Calculate the duration
		duration = end - start;
		// Output the duration in seconds
		std::cout << "(Deterministic)Time taken: " << duration.count() << " seconds" << std::endl;

		// Start the clock
		start = std::chrono::high_resolution_clock::now();

		double bestSse4 = runSplit(dataPoints, NUM_CENTROIDS);

		// Stop the clock
		end = std::chrono::high_resolution_clock::now();
		// Calculate the duration
		duration = end - start;
		// Output the duration in seconds
		std::cout << "(Split)Time taken: " << duration.count() << " seconds" << std::endl;

		std::cout << "(K-means)Best Sum-of-Squared Errors (SSE): " << bestSse1 << std::endl;
		std::cout << "(Repeated K-means)Best Sum-of-Squared Errors (SSE): " << bestSse5 << std::endl;
		std::cout << "(Random Swap)Best Sum-of-Squared Errors (SSE): " << bestSse2 << std::endl;
		std::cout << "(Deterministic Swap)Best Sum-of-Squared Errors (SSE): " << bestSse3 << std::endl;
		std::cout << "(Split)Best Sum-of-Squared Errors (SSE): " << bestSse4 << std::endl;

		return 0;
	}

	//subset stuff
	if (true) {
		std::cout << "Number of dimensions in the data: " << numDimensions << std::endl;

		std::vector<DataPoint> dataPoints = readDataPoints(DATA_FILENAME);
		std::vector <DataPoint> subSet;
		double obsoletes = 0.60; //subset size = 1 - obsoletes = 1 - 0,6 = 40%

		createSubSet(dataPoints, obsoletes);
		for (int i = 0; i < dataPoints.size(); ++i) {
			if (dataPoints[i].obsolete == false)
			{
				dataPoints[i].ogIndex = i;
				subSet.push_back(dataPoints[i]);
			}
		}

		// FullSet
		std::vector<DataPoint> centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);

		// Initial partition and SSE
		std::vector<int> initialPartition = optimalPartition(dataPoints, centroids);
		double initialSSE = calculateSSE(dataPoints, centroids, initialPartition);
		std::cout << "Initial Total Sum-of-Squared Errors (SSE): " << initialSSE << std::endl;

		std::pair<double, std::vector<int>> bestSse1 = runKMeans(dataPoints, MAX_ITERATIONS, centroids, true);
		std::cout << "(FullSet)Best Sum-of-Squared Errors (SSE): " << bestSse1.first << std::endl;

		//SubSet
		std::vector<DataPoint> subSetCentroids = generateRandomCentroids(NUM_CENTROIDS, subSet);

		// Initial partition and SSE using random centroids
		std::vector<int> subSetInitialPartition = optimalPartition(subSet, subSetCentroids);
		double subSetInitialSSE = calculateSSE(subSet, subSetCentroids, subSetInitialPartition);
		std::cout << "(SubSet) Initial Total Sum-of-Squared Errors (SSE): " << subSetInitialSSE << std::endl;

		std::pair<double, std::vector<int>> subSetBestSse1 = runKMeans(subSet, MAX_ITERATIONS, subSetCentroids, true);
		std::cout << "(SubSet)Best Sum-of-Squared Errors (SSE): " << subSetBestSse1.first << std::endl;

		//Check for stability = are points that form a cluster in the sub set also in the same cluster in the full set?
		bool stab = checkStability(subSet, subSetBestSse1.second, bestSse1.second);
		std::cout << "Stability: " << (stab ? "yes" : "no") << std::endl;

		return 0;
	}

	//grid stuff (unfinished)
	if (false) {
		std::cout << "Number of dimensions in the data: " << numDimensions << std::endl;

		// Read data points
		std::vector<DataPoint> dataPoints = readDataPoints(DATA_FILENAME);
		// Generate and write centroids
		std::vector<DataPoint> centroids = generateRandomCentroids(NUM_CENTROIDS, dataPoints);
		std::vector<DataPoint> ogCentroids(centroids.begin(), centroids.end());

		// Initial partition and SSE using random centroids
		std::vector<int> initialPartition = optimalPartition(dataPoints, centroids);
		double initialSSE = calculateSSE(dataPoints, centroids, initialPartition);
		std::cout << "Initial Total Sum-of-Squared Errors (SSE): " << initialSSE << std::endl;
		double bestSse1 = runKMeans(dataPoints, MAX_ITERATIONS, centroids);
		std::cout << "(Randomized)Best Sum-of-Squared Errors (SSE): " << bestSse1 << std::endl;


		// Read data points
		std::vector<DataPoint> ogDataPoints = readDataPoints(DATA_FILENAME);

	}
}