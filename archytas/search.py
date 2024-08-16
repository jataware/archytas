import pdb
from typing import Literal
from archytas.tool_utils import tool
from serpapi import search


@tool()
class Search:
    def __init__(self, serpapi_api_key: str, results_per_page: int = 10, engine: str = 'google',
                 device: Literal['desktop', 'tablet', 'mobile'] = 'desktop'):
        self.serpapi_api_key = serpapi_api_key
        self.results_per_page = results_per_page
        self.engine = engine
        self.device = device

    @tool()
    def websearch(
        self,
        q: str,
        location: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        # nfpr: bool = False,
        # filter_similar: bool = False,

    ):
        ...


"""


q
Required
Parameter defines the query you want to search. You can use anything that you would use in a regular Google search. e.g. inurl:, site:, intitle:. We also support advanced search query parameters such as as_dt and as_eq. See the full list of supported advanced search query parameters.
Geographic Location

location
Optional
Parameter defines from where you want the search to originate. If several locations match the location requested, we'll pick the most popular one. Head to the /locations.json API if you need more precise control. The location and uule parameters can't be used together. It is recommended to specify location at the city level in order to simulate a real user’s search. If location is omitted, the search may take on the location of the proxy.

uule
Optional
Parameter is the Google encoded location you want to use for the search. uule and location parameters can't be used together.
Advanced Google Parameters

ludocid
Optional
Parameter defines the id (CID) of the Google My Business listing you want to scrape. Also known as Google Place ID.

lsig
Optional
Parameter that you might have to use to force the knowledge graph map view to show up. You can find the lsig ID by using our Local Pack API or Google Local API.
lsig ID is also available via a redirect Google uses within Google My Business.

kgmid
Optional
Parameter defines the id (KGMID) of the Google Knowledge Graph listing you want to scrape. Also known as Google Knowledge Graph ID. Searches with kgmid parameter will return results for the originally encrypted search parameters. For some searches, kgmid may override all other parameters except start, and num parameters.

si
Optional
Parameter defines the cached search parameters of the Google Search you want to scrape. Searches with si parameter will return results for the originally encrypted search parameters. For some searches, si may override all other parameters except start, and num parameters. si can be used to scrape Google Knowledge Graph Tabs.

ibp
Optional
Parameter is responsible for rendering layouts and expansions for some elements (e.g., gwp;0,7 to expand searches with ludocid for expanded knowledge graph).
Localization

google_domain
Optional
Parameter defines the Google domain to use. It defaults to google.com. Head to the Google domains page for a full list of supported Google domains.

gl
Optional
Parameter defines the country to use for the Google search. It's a two-letter country code. (e.g., us for the United States, uk for United Kingdom, or fr for France). Head to the Google countries page for a full list of supported Google countries.

hl
Optional
Parameter defines the language to use for the Google search. It's a two-letter language code. (e.g., en for English, es for Spanish, or fr for French). Head to the Google languages page for a full list of supported Google languages.

cr
Optional
Parameter defines one or multiple countries to limit the search to. It uses country{two-letter upper-case country code} to specify countries and | as a delimiter. (e.g., countryFR|countryDE will only search French and German pages). Head to the Google cr countries page for a full list of supported countries.

lr
Optional
Parameter defines one or multiple languages to limit the search to. It uses lang_{two-letter language code} to specify languages and | as a delimiter. (e.g., lang_fr|lang_de will only search French and German pages). Head to the Google lr languages page for a full list of supported languages.
Advanced Filters

tbs
Optional

(to be searched) parameter defines advanced search parameters that aren't possible in the regular query field. (e.g., advanced search for patents, dates, news, videos, images, apps, or text contents).

safe
Optional
Parameter defines the level of filtering for adult content. It can be set to active or off, by default Google will blur explicit content.

nfpr
Optional
Parameter defines the exclusion of results from an auto-corrected query when the original query is spelled wrong. It can be set to 1 to exclude these results, or 0 to include them (default). Note that this parameter may not prevent Google from returning results for an auto-corrected query if no other results are available.

filter
Optional
Parameter defines if the filters for 'Similar Results' and 'Omitted Results' are on or off. It can be set to 1 (default) to enable these filters, or 0 to disable these filters.
Search Type

tbm
Optional
(to be matched) parameter defines the type of search you want to do.
It can be set to:
(no tbm parameter): regular Google Search,
isch: Google Images API,
lcl - Google Local API
vid: Google Videos API,
nws: Google News API,
shop: Google Shopping API,
pts: Google Patents API,
or any other Google service.
Pagination

start
Optional
Parameter defines the result offset. It skips the given number of results. It's used for pagination. (e.g., 0 (default) is the first page of results, 10 is the 2nd page of results, 20 is the 3rd page of results, etc.).

Google Local Results only accepts multiples of 20(e.g. 20 for the second page results, 40 for the third page results, etc.) as the start value.

num
Optional
Parameter defines the maximum number of results to return. (e.g., 10 (default) returns 10 results, 40 returns 40 results, and 100 returns 100 results).
Serpapi Parameters

engine
Optional

Set parameter to google (default) to use the Google API engine.

device
Optional
Parameter defines the device to use to get the results. It can be set to desktop (default) to use a regular browser, tablet to use a tablet browser (currently using iPads), or mobile to use a mobile browser (currently using iPhones).

no_cache
Optional
Parameter will force SerpApi to fetch the Google results even if a cached version is already present. A cache is served only if the query and all parameters are exactly the same. Cache expires after 1h. Cached searches are free, and are not counted towards your searches per month. It can be set to false (default) to allow results from the cache, or true to disallow results from the cache. no_cache and async parameters should not be used together.

async
Optional
Parameter defines the way you want to submit your search to SerpApi. It can be set to false (default) to open an HTTP connection and keep it open until you got your search results, or true to just submit your search to SerpApi and retrieve them later. In this case, you'll need to use our Searches Archive API to retrieve your results. async and no_cache parameters should not be used together. async should not be used on accounts with Ludicrous Speed enabled.

api_key

Required
Parameter defines the SerpApi private key to use.

output
Optional
Parameter defines the final output you want. It can be set to json (default) to get a structured JSON of the results, or html to get the raw html retrieved.

    """


params = {
    "engine": "google",
    "q": "Fresh Bagels",
    "location": "Seattle-Tacoma, WA, Washington, United States",
    "hl": "en",
    "gl": "us",
    "google_domain": "google.com",
    "num": "10",
    "start": "10",
    "safe": "active",
    "api_key": "ca6c2005f257208fbe9fcdfaf59352f641f77840187822764d287bf2d09236d4"
}


results = search(params)
organic_results = results["organic_results"]

pdb.set_trace()


@tool()
class Search:
    def __init__(self, serp_api_key: str):
        """
        Search tool using SERP API

        Args:
            serp_api_key (str): The API key for the SERP API
        """
        self.serp_api_key = serp_api_key

    @tool()
    def search(self, query: str,): ...
