import requests


# call the function after starting the micro-service
def api_predict(domain):
    # call api service
    url = "http://0.0.0.0:4000/predict"
    params = {"domain": domain}
    # send get request and save the response as response object
    r = requests.get(url=url, params=params)
    # extract data in json format
    data = r.json()

    return data


if __name__ == "__main__":

    # example
    domains = ["www.secureworks.com", \
               "www1.undefinedratiotanks.nrw", \
               "reifpiitfsmetvf.ru", \
               "o3kxkf19ttvw1vw801m4fy09h.org", \
               "broaddrawnkidney.pm", \
               "dev.depoxywehkchr.ru"]

    results = []

    for domain in domains:
        results.append(api_predict(domain))

    for result in results:
        print(result)
