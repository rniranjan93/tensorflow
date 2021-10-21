#include<bits/stdc++.h>
using namespace std;
#pragma warning(disable:4996)
const int m = INT_MAX / 2;
vector<int> arr(5002, -m);
vector<int> prev3(5002, m);
vector<int> prev2(5002, m);
vector<int> prev1(5002, m);
vector<int> pres(5002, m);
int n;
int down(int index)
{
	if (index == 1)
	{
		if (arr[index] <= arr[index + 1])
			return arr[index + 1] - arr[index] + 1;
		return 0;
	}
	else
		if (index == n)
		{
			if (arr[index] <= arr[index - 1])
				return arr[index - 1] - arr[index] + 1;
			return 0;
		}
	int l = 0;
	if (arr[index] <= arr[index + 1])
		l += arr[index + 1] - arr[index] + 1;
	if (arr[index] <= arr[index - 1])
		l += arr[index - 1] - arr[index] + 1;
	return l;
}
int ddown(int index)
{
	int l = 0;
	if (index == 1)
	{
		if (arr[index] <= arr[index + 1])
			return arr[index + 1] - arr[index] + 1;
		return 0;
	}
	else
		if (index == 2)
		{
			if (arr[index] <= arr[index - 1])
				l += arr[index - 1] - arr[index]+1;
			if (arr[index] <= arr[index + 1])
				l += arr[index + 1] - arr[index]+1;
			return l;
		}
		else
		{
			int k=arr[index-1];
			if (arr[index - 2] <= arr[index - 1])
				k -= arr[index - 1] - arr[index - 2] + 1;
			if (arr[index] <= k)
				l += k - arr[index] + 1;
		}
	if (arr[index] <= arr[index + 1])
		l += arr[index + 1] - arr[index];
	return l;
}
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	cin >> n;
	for (int i = 1; i <= n; i++)
		cin >> arr[i];
	int length = 0;
	for (int i = 1; i <= n; i++)
	{
		length = (i - 1) / 2;
		for (int j = length; j >= 0; j--)
		{
			if (j == 0)
			{
				pres[j] = down(i);
			}
			else
			pres[j] = min(prev3[j-1]+down(i),prev2[j-1]+ddown(i));			
		}
		for (int j = length; j >= 0; j--)
		{
			prev3[j] = min(prev3[j], prev2[j]);
			prev2[j] = prev1[j];
			prev1[j] = pres[j];
		}
	}

	for (int i = 0; i < (n+1) / 2; i++)
	{
		cout << min(min(prev3[i],prev2[i]), min(pres[i],prev1[i]))<<' ';
	}
	return 0;
}
