#:package spaceorc.Z3Wrap@0.0.8

using Spaceorc.Z3Wrap.Core;
using Spaceorc.Z3Wrap.Expressions.Common;
using Spaceorc.Z3Wrap.Expressions.Logic;
using Spaceorc.Z3Wrap.Expressions.Numerics;

// Hardcoded test data (level 71)
var guesses = new Guess[]
{
    new([10,100,26,15,8,34,74,61,72,5,65,60], 0, 1),
    new([58,71,25,15,37,92,14,72,61,62,27,84], 0, 2),
    new([62,94,96,14,79,19,66,77,100,5,27,41], 0, 0),
    new([63,2,8,25,16,3,76,82,11,100,88,80], 0, 2),
    new([59,92,1,29,41,15,81,13,55,54,17,58], 1, 0),
    new([35,60,38,95,8,73,11,58,2,10,76,78], 0, 2),
    new([70,74,9,2,18,14,91,59,53,27,58,34], 0, 0),
    new([89,88,78,11,8,80,37,25,46,61,77,7], 2, 2),
    new([20,93,23,66,50,43,34,95,13,59,48,42], 1, 1),
    new([84,2,49,52,25,47,38,7,18,44,8,48], 0, 2),
    new([36,87,88,8,60,71,19,92,21,16,18,66], 0, 0),
    new([18,51,87,79,100,17,46,15,70,31,90,8], 1, 1),
    new([80,5,59,27,58,71,99,18,34,86,92,63], 0, 2),
    new([34,59,58,47,80,83,49,24,10,85,37,16], 0, 1),
    new([77,97,93,85,47,27,55,66,81,63,29,100], 0, 1),
    new([16,13,37,50,88,69,27,53,36,8,41,86], 0, 1),
    new([5,11,54,68,36,29,50,35,77,81,89,71], 0, 1),
    new([63,10,56,7,27,20,48,38,64,70,94,39], 0, 2),
    new([23,8,13,51,30,6,78,90,100,83,84,40], 0, 1),
    new([59,45,10,11,6,46,30,90,79,98,74,66], 1, 1),
    new([79,16,45,49,92,9,20,36,15,26,61,56], 0, 2),
    new([70,75,30,76,49,78,25,18,40,62,81,51], 0, 2),
    new([15,60,90,21,77,38,70,88,39,55,47,20], 0, 3),
    new([46,72,34,35,29,74,30,81,89,20,56,25], 1, 4),
    new([99,38,73,68,79,25,15,49,59,94,75,81], 0, 2),
    new([50,30,22,36,38,15,99,88,57,4,73,94], 0, 2),
    new([13,60,41,53,69,21,11,20,98,23,77,35], 0, 2),
    new([58,83,72,18,5,45,19,2,88,53,52,51], 0, 0),
    new([31,82,78,5,33,62,57,86,42,48,81,66], 0, 3),
    new([87,76,75,49,79,4,70,26,94,100,42,89], 0, 1),
    new([50,49,83,5,69,53,9,91,2,37,98,73], 0, 0),
    new([84,38,66,82,37,47,94,68,3,27,21,25], 0, 2),
    new([38,53,11,7,44,4,41,52,56,89,93,14], 0, 1),
    new([29,92,64,78,36,89,81,77,46,74,61,63], 0, 2),
    new([13,92,42,61,52,9,53,64,67,63,8,73], 0, 2),
    new([55,50,46,58,85,94,84,52,6,98,30,8], 0, 3),
    new([55,23,64,69,34,68,96,94,87,77,18,88], 0, 1),
    new([35,99,40,94,54,52,47,73,74,3,53,76], 0, 1),
    new([67,71,82,100,32,76,47,93,14,57,31,53], 0, 1),
};

int length = guesses[0].Numbers.Length;
int maxNum = 100;

// Numbers that are definitely not in the answer
var impossible = guesses
    .Where(g => g.Bulls == 0 && g.Cows == 0)
    .SelectMany(g => g.Numbers)
    .ToHashSet();

Console.WriteLine($"Guesses: {guesses.Length}, Length: {length}, Impossible: {impossible.Count}");

var solution = new long[length];
var sw = System.Diagnostics.Stopwatch.StartNew();

// Build sorted candidate list
var candidates = Enumerable.Range(1, maxNum).Where(n => !impossible.Contains(n)).ToList();
Console.WriteLine($"Candidates: {candidates.Count}");

using var context = new Z3Context();
using var scope = context.SetUp();

var X = Enumerable.Range(0, length).Select(i => context.IntConst($"x_{i}")).ToArray();
using var solver = context.CreateSolver();

// Basic constraints (added once)
foreach (var x in X)
{
    solver.Assert(x >= 1);
    solver.Assert(x <= maxNum);
    foreach (var imp in impossible)
        solver.Assert(x != imp);
}

// All distinct
solver.Assert(context.Distinct(X));

// Guess constraints (added once)
var one = context.Int(1);
var zero = context.Int(0);
foreach (var g in guesses)
{
    // Bulls: exact position match
    var bulls = context.Add(X.Select((x, i) => (x == g.Numbers[i]).Ite(one, zero)));
    solver.Assert(bulls == g.Bulls);

    // Total matches: for each unique number in guess, check if it appears anywhere in solution
    var uniqueNums = g.Numbers.Distinct().Where(n => !impossible.Contains(n)).ToList();
    var matchIndicators = uniqueNums.Select(num =>
    {
        var appearsAnywhere = context.Or(X.Select(x => x == num));
        return appearsAnywhere.Ite(one, zero);
    });
    // total matches = bulls + cows
    solver.Assert(context.Add(matchIndicators) == g.Bulls + g.Cows);
}

// Find lexicographically minimal solution
for (int pos = 0; pos < length; pos++)
{
    Console.Write($"Position {pos}... ");

    // Binary search for minimum
    long lo = 1, hi = maxNum, best = -1;
    while (lo <= hi)
    {
        long mid = (lo + hi) / 2;
        solver.Push();
        solver.Assert(X[pos] <= mid);

        if (solver.Check() == Z3Status.Satisfiable)
        {
            best = (long)solver.GetModel().GetIntValue(X[pos]);
            hi = mid - 1;
        }
        else
        {
            lo = mid + 1;
        }
        solver.Pop();
    }

    // Fix this position permanently
    solver.Assert(X[pos] == best);
    solution[pos] = best;
    Console.WriteLine($"{best}");
}

Console.WriteLine($"\nSolution found in {sw.Elapsed.TotalSeconds:F1}s:");
var answer = string.Join(" ", solution);
Console.WriteLine(answer);

// Verify
Console.WriteLine("\nVerifying...");
bool valid = guesses.All(g =>
{
    int bulls = solution.Zip(g.Numbers).Count(p => p.First == p.Second);
    var secretSet = solution.ToHashSet();
    int cows = g.Numbers.Where((n, i) => solution[i] != n && secretSet.Contains(n)).Count();
    return bulls == g.Bulls && cows == g.Cows;
});

Console.WriteLine(valid ? "VALID!" : "INVALID!");
Console.WriteLine($"\nAnswer: {answer}");

record Guess(int[] Numbers, int Bulls, int Cows);
