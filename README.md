#include<bits/stdc++.h>
using namespace std;
long long dp[3][300005];
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int n;
	int arr;
	long long x;
	cin >> n >> x;
	long long maxx = 0;
	for (int i = 1; i <= n; i++)
	{
		cin >> arr;
		dp[0][i] = max((long long)0,dp[0][i-1]);
		dp[1][i] = max((long long)0,max(dp[1][i - 1], dp[0][i-1])+x * arr);
		dp[2][i] = max((long long)0, max(dp[2][i - 1], dp[1][i - 1]) + arr);
		for (int j = 0; j < 3; j++)
		{
			maxx = max(dp[j][i], maxx);
		}
	}
	cout << maxx;
	return 0;
}
