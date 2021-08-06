#include<bits/stdc++.h>
using namespace std;
#define MOD 1000000007
int main()
{
	int n, m;
	int sum = -2;
	int temp;
	int temp1 = 0, temp2 = 2;
	//scanf("%d%d", &n, &m);
	cin >> n >> m;
	if (n > m)
	{
		swap(n, m);
	}
	int i = 0;
	while (i <n)
	{
		temp = temp2;
		temp2 = (temp1 + temp2)%MOD;
		temp1 = temp;
		i++;
	}
	sum += temp2;
	while (i < m)
	{
		temp = temp2;
		temp2 = (temp1 + temp2)%MOD;
		temp1 = temp;
		i++;
	}
	printf("%d", (sum + temp2) % MOD);
	return 0;
}

