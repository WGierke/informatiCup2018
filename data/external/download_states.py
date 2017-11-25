import requests

short_names = ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']
url = "http://www.spiketime.de/feiertagapi/feiertage/{name}/{year}"

for year in range(2014, 2019):
    for name in short_names:
        content = requests.get(url.format(**locals()), verify=False).text
        print(content)
        with open("vacations_{year}_{name}.json".format(**locals()), "w") as f:
            f.write(str(content))
