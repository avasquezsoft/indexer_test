"""
title: Tennis Doc RAG
author: Tritech Prime
version: 1.6
description: >
  Inyecta automáticamente contexto del código fuente indexado en cada conversación.
  Detecta repositorios y ramas mencionados en la pregunta. Rama por defecto: prod.
  Permite disparar indexación escribiendo "indexa org/repo [rama]" en el chat.
  Permite generar PDFs de la última respuesta del asistente.
"""

import base64
import os
import re
import requests

# Regex para detectar formato org/repo en cualquier parte del mensaje
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+")
# Regex para detectar rama explícita: "rama X" o "branch X"
_BRANCH_RE = re.compile(r"(?:rama|branch)\s+(\S+)", re.IGNORECASE)
# Regex para detectar solicitud de PDF
_PDF_RE = re.compile(
    r"(?:\b(?:genera?(?:r|me)?|crea?(?:r|me)?|descarga?(?:r|me)?|exporta?(?:r|me)?|guarda?(?:r|me)?|sacar|dame|mostra?(?:r|me)?|hazme|preparame|armame)\b).*?\bpdf\b",
    re.IGNORECASE,
)
# Regex para detectar solicitud de Markdown / archivo MD
_MD_RE = re.compile(
    r"(?:\b(?:genera?(?:r|me)?|crea?(?:r|me)?|descarga?(?:r|me)?|exporta?(?:r|me)?|guarda?(?:r|me)?|sacar|dame|mostra?(?:r|me)?|hazme|preparame|armame)\b).*?\b(?:markdown|md|\.md)\b",
    re.IGNORECASE,
)
# Regex para comando especial "grafo NombreClase"
_GRAPH_RE = re.compile(r"^\s*grafo\s+([A-Z][a-zA-Z0-9]*)\s*$", re.IGNORECASE)
# Regex para listar repositorios indexados
_REPOS_LIST_RE = re.compile(r"^\s*(repos|repositorios|listar\s+repos?)\s*$", re.IGNORECASE)
# Palabras que indican que el usuario busca implementación/código
_IMPL_KEYWORDS = re.compile(
    r"\b(m[oó]dulo|module|implementaci[oó]n|implementation|expl[íi]came|explica|c[oó]mo\s+funciona|queries?|sql|dao|repositorio|service|servicio|m[eé]todos?|clase|class|business\s+logic|l[oó]gica)\b",
    re.IGNORECASE,
)
# Extrae posibles nombres de módulo de un path tipo "foo/bar/module-name"
_MODULE_PATH_RE = re.compile(r"[\w-]+/[\w-]+/([\w-]+)")
# Detecta nombres de clases Java en la query (ej: ConsultarInventarioVtexServiceImpl)
_CLASS_NAME_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:Impl|Dao|Service|Repository|Mapper|Controller|Dto|Entity|Config|Util|Factory|Handler|Listener|Task|Job|Batch|Processor|Writer|Reader|Interceptor|Filter|Servlet|Endpoint|Client|Provider|Delegate|Adapter|Facade|Builder|Validator|Converter|Formatter|Parser|Renderer|Generator|Exporter|Importer|Scheduler|Monitor|Tracker|Logger|Auditor|Security|Session|Token|Resolver|Locator|Registry|Cache|Pool|Queue|Stack|Map|Tree|Graph|Node|Connection|Transaction|Context|Event|Message|Command|Query|Request|Response|Result|Source|Target|Reference|Wrapper|Proxy|Stub|Mock|Spy|Matcher|Verifier|Checker|Tester|Inspector|Reviewer|Normalizer|Sanitizer|Cleaner|Splitter|Joiner|Merger|Sorter|Ranker|Scorer|Evaluator|Calculator|Computer|Estimator|Predictor|Classifier|Clusterer|Finder|Searcher|Indexer|Extractor|Loader|Saver|Persister|Retriever|Updater|Deleter|Creator|Destroyer|Initializer|Finalizer|Activator|Deactivator|Enabler|Disabler|Opener|Closer|Starter|Stopper|Launcher|Runner|Executor|Invoker|Caller|Dispatcher|Router|Balancer|Distributor|Allocator|Assigner|Configurer|Setter|Getter|Accessor|Mutator|Builder|Maker|Producer|Consumer|Subscriber|Publisher|Emitter|Receiver|Sender|Transmitter|Broadcaster|Multicaster|Peer|Host|Guest|Client|Server|Master|Slave|Primary|Secondary|Main|Auxiliary|Helper|Utility|Tool|Kit|Lib|Api|Sdk|Cli|Gui|Ui|Web|Rest|Soap|Grpc|Graphql|Websocket|Socket|Port|Channel|Pipe|Stream|Flow|Pipeline|Chain|Sequence|Series|Batch|Bundle|Pack|Package|Module|Component|Part|Piece|Section|Segment|Fragment|Chunk|Block|Unit|Item|Element|Member|Field|Property|Attribute|Parameter|Argument|Option|Setting|Configuration|Preference|Policy|Rule|Strategy|Pattern|Template|Schema|Model|Blueprint|Plan|Design|Layout|Structure|Framework|Platform|System|Engine|Kernel|Core|Base|Root|Foundation|Layer|Tier|Level|Stage|Phase|Step|Action|Operation|Process|Procedure|Routine|Function|Method|Subroutine|Macro|Script|Program|Application|App|Service|Microservice|Daemon|Agent|Bot|Worker|Thread|Task|Job|Process|Instance|Object|Bean|Component|Module|Plugin|Extension|Addon|Integration|Connector|Bridge|Gateway|Proxy|Tunnel|Vpn|Firewall|Shield|Guard|Watcher|Observer|Listener|Monitor|Sensor|Detector|Analyzer|Scanner|Probe|Tracer|Tracker|Logger|Recorder|Collector|Aggregator|Compiler|Interpreter|Translator|Converter|Adapter|Wrapper|Facade|Delegate|Proxy|Stub|Mock|Fake|Dummy|Spy|Captor|Matcher|Verifier|Asserter|Checker|Tester|Inspector|Reviewer|Auditor|Validator|Sanitizer|Normalizer|Formatter|Parser|Lexer|Tokenizer|Splitter|Joiner|Merger|Combiner|Mixer|Blender|Fusion|Integration|Unification|Composition|Aggregation|Association|Relation|Link|Reference|Pointer|Handle|Descriptor|Metadata|Annotation|Tag|Label|Marker|Flag|Indicator|Signal|Trigger|Event|Notification|Alert|Warning|Error|Exception|Failure|Fault|Defect|Bug|Issue|Ticket|Case|Scenario|Story|Epic|Saga|Transaction|Session|Conversation|Dialog|Chat|Message|Mail|Letter|Note|Memo|Document|Record|Entry|Log|History|Audit|Trail|Trace|Track|Path|Route|Way|Road|Street|Avenue|Boulevard|Highway|Freeway|Motorway|Turnpike|Tollway|Expressway|Parkway|Driveway|Walkway|Pathway|Trail|Track|Trace|Line|Route|Circuit|Loop|Ring|Circle|Cycle|Orbit|Spiral|Helix|Coil|Twist|Turn|Bend|Curve|Arc|Bow|Arch|Vault|Dome|Canopy|Cover|Roof|Ceiling|Top|Peak|Summit|Crest|Ridge|Range|Chain|Series|Sequence|Succession|Progression|Line|Row|Rank|File|Column|Pillar|Post|Pole|Shaft|Stem|Trunk|Stock|Root|Base|Foundation|Foot|Bottom|Floor|Ground|Soil|Earth|Land|Terrain|Territory|Country|Nation|State|Province|Region|Zone|Area|District|Quarter|Neighborhood|Vicinity|Locality|Place|Spot|Site|Position|Location|Situation|Station|Post|Base|Camp|Settlement|Colony|Outpost|Hub|Center|Core|Heart|Middle|Midst|Interior|Inside|Within|Inner|Internal|Inward|Central|Focal|Key|Main|Primary|Principal|Chief|Leading|First|Prime|Premier|Head|Top|Upper|Higher|Superior|Supreme|Ultimate|Final|Last|Ultimate|Extreme|Utmost|Maximum|Maximal|Peak|Top|Crest|Summit|Apex|Vertex|Zenith|Acme|Pinnacle|Climax|Culmination|Crown|Cap|Tip|Point|Dot|Spot|Speck|Grain|Particle|Atom|Molecule|Cell|Unit|Element|Component|Constituent|Ingredient|Factor|Aspect|Feature|Characteristic|Property|Attribute|Quality|Trait|Mark|Sign|Indication|Evidence|Proof|Token|Symbol|Emblem|Badge|Banner|Flag|Standard|Colors|Insignia|Regalia|Trappings|Gear|Equipment|Apparatus|Instrument|Tool|Implement|Device|Gadget|Contraption|Machine|Engine|Motor|Mechanism|Appliance|Utensil|Vessel|Container|Receptacle|Holder|Carrier|Bearer|Conveyor|Transporter|Vehicle|Vessel|Craft|Ship|Boat|Ferry|Barge|Yacht|Cruiser|Liner|Tanker|Carrier|Freighter|Cargo|Hauler|Tractor|Truck|Van|Bus|Coach|Car|Automobile|Vehicle|Ride|Wheel|Cycle|Bike|Motorcycle|Scooter|Moped|Segway|Hoverboard|Skateboard|Rollerblade|Ski|Snowboard|Sled|Sleigh|Carriage|Chariot|Wagon|Cart|Buggy|Coach|Sedan|Limousine|Cab|Taxi|Hack|Jitney|Rickshaw|Tuk-tuk|Bicycle|Tricycle|Unicycle|Quadracycle|Velomobile|Recumbent|Tandem|Surrey|Buckboard|Dogcart|Trap|Gig|Cabriolet|Landau|Brougham|Berlin|Coupe|Sedan|Saloon|Hatchback|Station|Wagon|Estate|Minivan|Minibus|Microbus|Van|Camper|RV|Motorhome|Trailer|Caravan|Fifth|Wheel|Popup|Tent|Teardrop|Airstream|Bus|Coach|Autobus|Omnibus|Trolley|Tram|Streetcar|Trolleybus|Trackless|Trolley|Subway|Metro|Underground|Tube|Rail|Train|Locomotive|Engine|Railcar|Carriage|Coach|Wagon|Caboose|Boxcar|Flatcar|Tank|Car|Hopper|Gondola|Reefer|Stock|Auto|Rack|Intermodal|Well|Container|Bulkhead|Centerbeam|Covered|Open|Skeleton|Depressed|Clearance|Lowboy|Stretch|Double|Triple|Extendable|Drop|Deck|Step|Deck|RGN|Removable|Gooseneck|Platform|Pallet|Skid|Stillage|Cage|Crate|Box|Case|Carton|Package|Parcel|Packet|Pouch|Bag|Sack|Barrel|Drum|Keg|Cask|Firkin|Hogshead|Puncheon|Tierce|Pipe|Butt|Barrel|Cask|Vat|Tun|Tank|Cistern|Reservoir|Basin|Pool|Pond|Lake|Lagoon|Loch|Fjord|Inlet|Cove|Bay|Gulf|Bight|Sound|Strait|Channel|Passage|Narrows|Throat|Gap|Break|Ravine|Gorge|Canyon|Valley|Dale|Vale|Glen|Hollow|Depression|Basin|Bowl|Crater|Caldera|Cavity|Hole|Opening|Aperture|Vent|Port|Gate|Door|Entry|Entrance|Access|Approach|Way|Path|Route|Road|Street|Avenue|Boulevard|Drive|Lane|Alley|Court|Place|Terrace|Plaza|Square|Circle|Loop|Crescent|Heights|Hills|Ridge|Peak|Summit|Crest|View|Outlook|Overlook|Vista|Panorama|Scene|Sight|Spectacle|Display|Show|Exhibition|Exhibit|Presentation|Demonstration|Performance|Production|Entertainment|Amusement|Recreation|Pastime|Hobby|Diversion|Distraction|Relaxation|Rest|Repose|Respite|Relief|Ease|Comfort|Solace|Consolation|Support|Aid|Help|Assistance|Service|Favor|Kindness|Goodwill|Benevolence|Charity|Philanthropy|Humanitarianism|Altruism|Selflessness|Generosity|Liberality|Munificence|Magnanimity|Nobility|Honor|Integrity|Probity|Rectitude|Righteousness|Virtue|Goodness|Morality|Ethics|Principles|Standards|Values|Beliefs|Convictions|Tenets|Doctrines|Dogmas|Creeds|Catechisms|Canons|Laws|Rules|Regulations|Statutes|Ordinances|Decrees|Edicts|Proclamations|Declarations|Announcements|Notifications|Notices|Bulletins|Communications|Messages|Memoranda|Minutes|Records|Archives|Annals|Chronicles|Histories|Accounts|Narratives|Stories|Tales|Legends|Sagas|Myths|Fables|Parables|Allegories|Metaphors|Similes|Analogies|Comparisons|Contrasts|Juxtapositions|Oppositions|Differences|Distinctions|Variations|Divergences|Deviations|Departures|Digressions|Tangents|Excursions|Forays|Sorties|Sallies|Raids|Incursions|Invasions|Attacks|Assaults|Offensives|Drives|Pushes|Thrusts|Lunges|Strikes|Hits|Blows|Knocks|Thumps|Bangs|Slams|Cracks|Snaps|Pops|Clicks|Clacks|Clinks|Chinks|Tinks|Jingles|Rings|Tolls|Peals|Chimes|Carillons|Knells|Bongs|Booms|Roars|Thunders|Claps|Crackles|Rustles|Whispers|Murmurs|Mumbles|Mutters|Grumbles|Growls|Snarls|Barks|Yaps|Yips|Yelps|Howls|Wails|Moans|Groans|Sighs|Gasps|Pants|Puffs|Huffs|Blasts|Gusts|Winds|Breezes|Zephyrs|Drafts|Currents|Streams|Flows|Floods|Torrents|Rivers|Brooks|Creeks|Rills|Runs|Springs|Wells|Fountains|Geysers|Spouts|Jets|Sprays|Mists|Fogs|Clouds|Hazes|Smogs|Vapors|Steams|Smokes|Fumes|Exhausts|Emissions|Discharges|Releases|Emissions|Effluents|Outflows|Runoffs|Spills|Leaks|Seepages|Oozings|Drips|Drops|Droplets|Beads|Blobs|Globs|Lumps|Chunks|Hunks|Blocks|Bricks|Cakes|Bars|Rods|Sticks|Strips|Bands|Ribbons|Tapes|Films|Membranes|Skins|Hides|Pelts|Furs|Coats|Jackets|Vests|Waistcoats|Shirts|Blouses|Tops|Tees|Tanks|Camisoles|Bras|Bandeaus|Bustiers|Corsets|Basques|Bodies|TeddIES|Negligees|Nightgowns|Nighties|Nightshirts|Pajamas|Pyjamas|Jammies|Sleepers|Onesies|Rompers|Jumpsuits|Coveralls|Overalls|Dungarees|Jeans|Trousers|Pants|Slacks|Chinos|Khakis|Cords|Cords|Shorts|Bermudas|Cutoffs|Trunks|Boxers|Briefs|Undies|Underpants|Panties|Thongs|G-strings|Jockstraps|Athletic|Supporters|Cups|Guards|Protectors|Shields|Pads|Armor|Plates|Vests|Helmets|Hats|Caps|Bonnets|Berets|Beanies|Toques|Tam|o'|Shanters|Balmorals|Glengarries|Caubeens|Busbies|Shakos|Kepis|Forage|Caps|Peaked|Caps|Service|Caps|Field|Caps|Patrol|Caps|Baseball|Caps|Snapbacks|Trucker|Caps|Bucket|Hats|Sun|Hats|Sombreros|Panamas|Fedora|Trilby|Homburg|Porkpie|Bowler|Derby|Top|Hats|Stovepipes|Coachmen|Hats|Astrakhan|Bearskins|Busby|Shako|Pickelhaube|Spiked|Helmet|Sallet|Burgonet|Armet|Close|Helmet|Great|Helmet|Grand|Helmet|Frog-|mouth|Helmet|Bascinet|Cervelliere|Nasal|Helmet|Spangenhelm|Lamellar|Helmet|Scale|Helmet|Lamellar|Corselet|Brigandine|Coat|of|Plates|Jack|of|Plates|Transitional|Cuirass|Breastplate|Backplate|Fauld|Tasset|Culet|Plackart|Besagew|Rerebrace|Vambrace|Gauntlet|Couter|Spaulder|Pauldron|Gorget|Buffe|Falling|Buff|Bevor|Chin|Guard|Ventail|Aventail|Camail|Standard|Gardbrace|Pasguard|Grandguard|Wrapper|Buff|Coat|Doublet|Pourpoint|Aketon|Gambeson|Arming|Doublet|Hauberk|Byrnie|Lorica|Hamata|Lorica|Squamata|Lorica|Segmentata|Laminar|Armor|Laminar|Cuisses|Poleyns|Greaves|Sabatons|Sollaret|Solleret))\b")

# Detecta palabras que indican implementación concreta (Impl, Dao, Service, etc.)
_IMPL_SUFFIX_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:Impl|Dao|Service|Repository|Mapper|Controller))\b")


class Filter:
    def __init__(self):
        self.name = "Tennis Doc RAG"
        self.valves = self.Valves()
        print("[TennisDoc RAG] Filter cargado correctamente")

    class Valves:
        def __init__(self):
            self.indexer_url = "http://indexer:8001"
            self.limit = 20
            self.default_branch = "prod"
            self.api_key = os.environ.get("INDEXER_API_KEY", "")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.valves.api_key:
            h["Authorization"] = f"Bearer {self.valves.api_key}"
        return h

    def _enrich_query(self, query: str) -> str:
        """Enriquece la query del usuario con keywords técnicas para mejorar retrieval."""
        if not _IMPL_KEYWORDS.search(query):
            return query

        enrichment = []

        # Extraer posible nombre de módulo de paths mencionados
        module_match = _MODULE_PATH_RE.search(query)
        if module_match:
            enrichment.append(module_match.group(1).replace("-", " "))

        # Añadir keywords técnicas según lo que parece buscar
        lower = query.lower()
        if any(k in lower for k in ("query", "queries", "sql", "jpql", "select", "insert", "update")):
            enrichment.extend(["SQL query", "database", "DAO", "implementation"])
        if any(k in lower for k in ("módulo", "modulo", "module", "implementación", "implementation", "explícame", "explica", "cómo funciona")):
            enrichment.extend(["Java class", "implementation", "methods", "business logic", "DAO", "service"])
        if any(k in lower for k in ("dao", "repositorio", "repository")):
            enrichment.extend(["DAO", "implementation", "database queries"])

        if enrichment:
            return f"{query} {' '.join(enrichment)}"
        return query

    def _fetch_context(self, repo: str | None, branch: str | None, query: str) -> list[dict]:
        """Ejecuta el pipeline completo de búsqueda (vectorial + grafo + entidades)."""
        all_results: list[dict] = []
        seen_keys: set = set()

        def _add_result(r: dict, tag: str = ""):
            key = (r.get("file_path"), r.get("text", "")[:120])
            if key in seen_keys:
                for existing in all_results:
                    if (existing.get("file_path"), existing.get("text", "")[:120]) == key:
                        existing["score"] = max(existing.get("score", 0), r.get("score", 0))
                        if tag and tag not in str(existing.get("source", "")):
                            existing["source"] = f"{existing.get('source', 'unknown')}+{tag}"
                        break
                return
            seen_keys.add(key)
            if tag:
                r["source"] = f"{r.get('source', 'unknown')}+{tag}"
            all_results.append(r)

        search_query = self._enrich_query(query)

        # 1) Búsqueda aumentada: archivos completos
        try:
            payload = {
                "query": search_query,
                "repo": repo,
                "branch": branch,
                "max_files": 10,
                "vector_limit": 50,
            }
            resp = requests.post(
                f"{self.valves.indexer_url}/search-augmented",
                json={k: v for k, v in payload.items() if v is not None},
                headers=self._headers(),
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", []):
                _add_result(r, "augmented")
            print(f"[TennisDoc RAG] Búsqueda aumentada: {len(data.get('results', []))} chunks de {data.get('files_fetched', 0)} archivos")
        except Exception as e:
            print(f"[TennisDoc RAG] Error en /search-augmented: {e}")

        # 2) Búsqueda grafo: entidades y vecinos
        try:
            payload = {
                "query": search_query,
                "repo": repo,
                "branch": branch,
                "limit": self.valves.limit,
                "graph_depth": 2,
            }
            resp = requests.post(
                f"{self.valves.indexer_url}/search-graph",
                json={k: v for k, v in payload.items() if v is not None},
                headers=self._headers(),
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", []):
                _add_result(r, "graph")
            print(f"[TennisDoc RAG] Búsqueda grafo: {len(data.get('results', []))} resultados")
        except Exception as e:
            print(f"[TennisDoc RAG] Error en /search-graph: {e}")

        # 3) Si la query menciona nombres de clase, traer la entidad del grafo y sus relaciones directas
        class_names = _CLASS_NAME_RE.findall(query)
        if class_names and repo:
            for class_name in class_names[:3]:
                try:
                    resp = requests.get(
                        f"{self.valves.indexer_url}/graph/entity/{class_name}",
                        params={"repo": repo, "branch": branch or self.valves.default_branch},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        ent = resp.json()
                        if ent.get("code"):
                            _add_result({
                                "score": 1.0,
                                "repo": repo,
                                "branch": branch or self.valves.default_branch,
                                "file_path": ent.get("file_path", ""),
                                "language": "java",
                                "text": ent["code"],
                                "ast_name": ent.get("name", ""),
                                "ast_type": ent.get("type", ""),
                                "ast_signature": ent.get("signature", ""),
                                "source": "graph_entity",
                            }, "entity")
                            print(f"[TennisDoc RAG] Entidad grafo añadida: {class_name}")
                        for rel in ent.get("relations", [])[:5]:
                            target_name = rel.get("target_name")
                            if not target_name:
                                continue
                            try:
                                tresp = requests.get(
                                    f"{self.valves.indexer_url}/graph/entity/{target_name}",
                                    params={"repo": repo, "branch": branch or self.valves.default_branch},
                                    timeout=10,
                                )
                                if tresp.status_code == 200:
                                    tent = tresp.json()
                                    if tent.get("code"):
                                        _add_result({
                                            "score": 0.95,
                                            "repo": repo,
                                            "branch": branch or self.valves.default_branch,
                                            "file_path": tent.get("file_path", ""),
                                            "language": "java",
                                            "text": tent["code"],
                                            "ast_name": tent.get("name", ""),
                                            "ast_type": tent.get("type", ""),
                                            "ast_signature": tent.get("signature", ""),
                                            "source": "graph_relation",
                                        }, "relation")
                            except Exception as te:
                                print(f"[TennisDoc RAG] Error trayendo relación {target_name}: {te}")
                    else:
                        # Fallback: buscar archivo por nombre en el índice y hacer fetch
                        resp = requests.get(
                            f"{self.valves.indexer_url}/debug/files-indexed",
                            params={"repo": repo, "branch": branch or self.valves.default_branch},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        files = resp.json().get("files", [])
                        matches = [f for f in files if class_name in f]
                        for file_path in matches[:2]:
                            try:
                                fetch_resp = requests.post(
                                    f"{self.valves.indexer_url}/fetch-file",
                                    json={"repo": repo, "file_path": file_path, "branch": branch or self.valves.default_branch},
                                    headers=self._headers(),
                                    timeout=10,
                                )
                                fetch_resp.raise_for_status()
                                fdata = fetch_resp.json()
                                _add_result({
                                    "score": 1.0,
                                    "repo": repo,
                                    "branch": branch or self.valves.default_branch,
                                    "file_path": file_path,
                                    "language": "java",
                                    "text": fdata.get("content", ""),
                                    "source": "fetch",
                                }, "fetch")
                                print(f"[TennisDoc RAG] Fetch directo: {file_path}")
                            except Exception as fe:
                                print(f"[TennisDoc RAG] Error fetch {file_path}: {fe}")
                except Exception as ce:
                    print(f"[TennisDoc RAG] Error buscando clase {class_name}: {ce}")

        return all_results

    def inlet(self, body: dict, user: dict = None) -> dict:
        """Se ejecuta ANTES de enviar los mensajes al LLM."""
        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            last_msg = messages[-1]
            if last_msg.get("role") != "user":
                return body

            # El contenido puede ser string o lista (si tiene attachments)
            raw_content = last_msg.get("content", "")
            if isinstance(raw_content, list):
                # Extraer solo el texto del primer elemento
                query = str(raw_content[0].get("text", "")).strip() if raw_content else ""
            else:
                query = str(raw_content).strip()

            if not query:
                return body

            print(f"[TennisDoc RAG] Query recibida: {query[:80]}...")

            # ── Comando especial: indexar un repo desde el chat ──
            lower_q = query.lower()
            index_commands = ("indexa ", "indexar ", "reindexa ", "reindexar ", "index ")
            if lower_q.startswith(index_commands):
                args = query.split(" ", 1)[1].strip() if " " in query else ""
                parts = args.split()
                repo = parts[0] if parts else ""
                branch = parts[1] if len(parts) > 1 else "HEAD"
                if repo:
                    print(f"[TennisDoc RAG] Comando index detectado: {repo} @ {branch}")
                    return self._trigger_index(body, repo, branch)

            # ── Generar Markdown con contexto de chunks ──
            if _MD_RE.search(query):
                repo, branch = self._extract_repo_branch(query)
                return self._generate_markdown(body, repo, branch, query)

            # ── Generar PDF de la última respuesta del asistente ──
            if _PDF_RE.search(query):
                repo, branch = self._extract_repo_branch(query)
                return self._generate_pdf(body, repo, branch)

            # ── Comando especial: mostrar grafo de una clase ──
            graph_match = _GRAPH_RE.search(query)
            if graph_match:
                class_name = graph_match.group(1)
                return self._show_graph(body, class_name, repo, branch)

            # ── Comando especial: listar repos indexados ──
            if _REPOS_LIST_RE.search(query):
                return self._list_repos(body)

            # ── RAG automático: detectar repo/rama y buscar contexto ──
            repo, branch = self._extract_repo_branch(query)
            print(f"[TennisDoc RAG] Buscando contexto | repo={repo} | branch={branch} | enriched_query={self._enrich_query(query)[:100]}...")

            deduped = self._fetch_context(repo, branch, query)

            if deduped:
                context = self._build_context(deduped)
                scope = f"Repo: {repo} | Rama: {branch}" if repo else f"Todas las ramas (filtro: {branch})"
                system_msg = {
                    "role": "system",
                    "content": (
                        "Eres un asistente técnico especializado en el código fuente de la organización. "
                        f"Ámbito de búsqueda: {scope}. "
                        "Responde ÚNICAMENTE basándote en el siguiente contexto del código. "
                        "El contexto incluye relaciones de grafo (herencia, implementaciones, métodos, campos, inyecciones de dependencias). "
                        "IMPORTANTE: los fragmentos pueden contener el código Java junto con las queries SQL referenciadas (marcadas como '-- Referenced SQL:'). "
                        "Busca en TODOS los fragmentos: firmas de métodos, implementaciones, queries SQL/JPQL, lógica de negocio y mapeo de entidades. "
                        "Si una clase extiende o implementa otra, menciona la jerarquía completa. "
                        "Si la respuesta no está en el contexto, indica que no tienes información suficiente.\n\n"
                        f"{context}"
                    ),
                }
                messages.insert(-1, system_msg)
                body["messages"] = messages

        except Exception as e:
            print(f"[TennisDoc RAG] ERROR en inlet: {e}")
            # En caso de error, devolvemos el body sin modificar para no romper el chat

        return body

    def _extract_repo_branch(self, query: str):
        """Extrae repo (org/repo) y rama del query."""
        repos = _REPO_RE.findall(query)
        repo = repos[0] if repos else None
        branch_match = _BRANCH_RE.search(query)
        branch = branch_match.group(1) if branch_match else self.valves.default_branch
        return repo, branch

    def _trigger_index(self, body: dict, repo: str, branch: str = "HEAD") -> dict:
        """Dispara la indexación y reemplaza el mensaje del usuario con una confirmación."""
        try:
            payload = {"repo": repo, "branch": branch}
            resp = requests.post(
                f"{self.valves.indexer_url}/index",
                json=payload,
                headers=self._headers(),
                timeout=5,
            )
            data = resp.json()
            body["messages"][-1]["content"] = (
                f"[Sistema interno] He iniciado la indexación de `{repo}` @ `{branch}`. "
                f"Estado: {data.get('status', 'ok')}. "
                f"Por favor espera unos minutos y luego haz tu pregunta sobre ese repositorio."
            )
            print(f"[TennisDoc RAG] Indexación iniciada: {repo} @ {branch}")
        except Exception as e:
            body["messages"][-1]["content"] = (
                f"[Sistema interno] Error al iniciar indexación de `{repo}` @ `{branch}`: {e}"
            )
            print(f"[TennisDoc RAG] ERROR indexando: {e}")
        return body

    def _generate_markdown(self, body: dict, repo: str | None, branch: str | None, query: str) -> dict:
        """Modo Markdown: recupera contexto y pide al LLM que responda en formato Markdown."""
        try:
            print(f"[TennisDoc RAG] Modo Markdown | repo={repo} | branch={branch}")
            deduped = self._fetch_context(repo, branch, query)

            if not deduped:
                body["messages"][-1]["content"] = (
                    "No encontré chunks para responder. Intenta con una pregunta más específica o verifica que el repo esté indexado."
                )
                return body

            context = self._build_context(deduped)
            scope = f"Repo: {repo} | Rama: {branch}" if repo else f"Todas las ramas (filtro: {branch})"

            system_msg = {
                "role": "system",
                "content": (
                    "Eres un asistente técnico especializado en el código fuente de la organización. "
                    f"Ámbito de búsqueda: {scope}. "
                    "Responde ÚNICAMENTE basándote en el siguiente contexto del código. "
                    "El contexto incluye relaciones de grafo (herencia, implementaciones, métodos, campos, inyecciones de dependencias). "
                    "IMPORTANTE: los fragmentos pueden contener el código Java junto con las queries SQL referenciadas (marcadas como '-- Referenced SQL:'). "
                    "Busca en TODOS los fragmentos: firmas de métodos, implementaciones, queries SQL/JPQL, lógica de negocio y mapeo de entidades. "
                    "Si una clase extiende o implementa otra, menciona la jerarquía completa. "
                    "FORMATO REQUERIDO: responde ÚNICAMENTE en Markdown bien estructurado. "
                    "Usa tablas para listar entidades, bullets para detalles y bloques de código para snippets. "
                    "No añadas introducciones como 'A continuación...', ve directo al contenido. "
                    "Si la respuesta no está en el contexto, indica que no tienes información suficiente.\n\n"
                    f"{context}"
                ),
            }

            messages = body.get("messages", [])
            messages.insert(-1, system_msg)
            body["messages"] = messages
            print(f"[TennisDoc RAG] Contexto Markdown inyectado: {len(deduped)} fragmentos")
        except Exception as e:
            print(f"[TennisDoc RAG] ERROR en modo Markdown: {e}")
        return body

    def _generate_pdf(self, body: dict, repo: str, branch: str) -> dict:
        """Genera un PDF de la última respuesta del asistente y la ofrece como descarga."""
        try:
            messages = body.get("messages", [])
            # Buscar la última respuesta del asistente
            assistant_msg = None
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "assistant":
                    assistant_msg = msg
                    break

            if not assistant_msg:
                body["messages"][-1]["content"] = "No hay una respuesta previa del asistente para convertir a PDF."
                return body

            content = assistant_msg.get("content", "")
            title = f"Respuesta_{repo.replace('/', '_')}" if repo else "Respuesta"

            resp = requests.post(
                f"{self.valves.indexer_url}/pdf",
                json={"title": title, "content": content, "repo": repo, "branch": branch},
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            pdf_bytes = resp.content
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

            download_link = f'<a href="data:application/pdf;base64,{pdf_b64}" download="{title}.pdf">📄 Descargar PDF</a>'

            body["messages"][-1]["content"] = (
                f"He generado el PDF con la respuesta anterior. "
                f"Haz clic para descargarlo:\n\n{download_link}"
            )
            print(f"[TennisDoc RAG] PDF generado: {title}.pdf ({len(pdf_bytes)} bytes)")
        except Exception as e:
            body["messages"][-1]["content"] = f"[Sistema] Error generando PDF: {e}"
            print(f"[TennisDoc RAG] ERROR generando PDF: {e}")
        return body

    def _show_graph(self, body: dict, class_name: str, repo: str | None, branch: str | None) -> dict:
        """Muestra las relaciones de una entidad del grafo."""
        try:
            params = {"repo": repo, "branch": branch}
            resp = requests.get(
                f"{self.valves.indexer_url}/graph/entity/{class_name}",
                params={k: v for k, v in params.items() if v is not None},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            relations = data.get("relations", [])
            lines = [f"## Grafo de dependencias: {class_name}", ""]
            lines.append(f"**Tipo:** {data.get('type', 'Unknown')} | **Archivo:** `{data.get('file_path', 'N/A')}`")
            if data.get('signature'):
                lines.append(f"**Firma:** `{data['signature']}`")
            lines.append("")
            if relations:
                lines.append("### Relaciones directas:")
                for rel in relations:
                    lines.append(f"- **{rel['rel_type']}** → `{rel['target_name']}` ({rel['target_type']})")
            else:
                lines.append("Sin relaciones directas indexadas.")
            body["messages"][-1]["content"] = "\n".join(lines)
            print(f"[TennisDoc RAG] Grafo mostrado para {class_name}")
        except Exception as e:
            body["messages"][-1]["content"] = f"No pude obtener el grafo para `{class_name}`. Error: {e}"
            print(f"[TennisDoc RAG] ERROR mostrando grafo: {e}")
        return body

    def _list_repos(self, body: dict) -> dict:
        """Consulta al indexer la lista de repos indexados y la muestra en el chat."""
        try:
            resp = requests.get(
                f"{self.valves.indexer_url}/repos",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            repos = data.get("repos", [])
            if repos:
                lines = ["## Repositorios indexados 🗂️", ""]
                for r in repos:
                    lines.append(f"- `{r}`")
                lines.append("")
                lines.append("Para consultar uno, escribí: `del repo org/repo-name ...`")
            else:
                lines = ["No hay repositorios indexados todavía.", "", "Indexá uno con: `indexa org/repo [rama]`"]
            body["messages"][-1]["content"] = "\n".join(lines)
            print(f"[TennisDoc RAG] Listado de repos: {len(repos)} repos")
        except Exception as e:
            body["messages"][-1]["content"] = f"[Sistema] Error listando repos: {e}"
            print(f"[TennisDoc RAG] ERROR listando repos: {e}")
        return body

    def _build_context(self, results: list) -> str:
        """Formatea los chunks recuperados para el prompt, incluyendo metadata de grafo."""
        parts = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            file_path = r.get("file_path", "unknown")
            repo = r.get("repo", "unknown")
            branch = r.get("branch", "")
            text = r.get("text", "")[:12000]
            ast_type = r.get("ast_type", "")
            ast_name = r.get("ast_name", "")
            ast_signature = r.get("ast_signature", "")
            source = r.get("source", "unknown")
            branch_info = f" | Rama: {branch}" if branch else ""
            meta_parts = [f"Score: {score:.3f} | Source: {source} | Repo: {repo}{branch_info} | Archivo: {file_path}"]
            if ast_name:
                meta_parts.append(f"Entidad: {ast_type} {ast_name}")
            if ast_signature:
                meta_parts.append(f"Firma: {ast_signature}")
            parts.append(
                f"[Fragmento {i}] {' | '.join(meta_parts)}\n```\n{text}\n```"
            )
        return "\n\n---\n\n".join(parts)
