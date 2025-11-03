<body>
 <div class="container">
  <header>
   <h1>
    SQSE AEAD – GPU-beschleunigte Authentifizierte Verschlüsselung mit Sub-Quantum State Encryption
   </h1>
   <p>
    <strong>
     Version:
    </strong>
    1.0
   </p>
   <p>
    <strong>
     Veröffentlichungsdatum:
    </strong>
    2025-11-03
   </p>
   <p>
    <strong>
     Autor(en):
    </strong>
    Mermaid Architect AI
   </p>
   <p>
    <strong>
     Lizenz:
    </strong>
    <a href="https://github.com/RalfKrummel/SQSE-AEAD/blob/main/LICENSE" target="_blank">
     MIT License
    </a>
   </p>
  </header>
  <section>
   <h2>
    2. Executive Summary
   </h2>
   <p>
    Das Projekt "SQSE AEAD" stellt ein hochperformantes und sicheres Verschlüsselungssystem dar, das auf der innovativen "Sub-Quantum State Encryption" (SQSE) basiert und eine GPU-beschleunigte Keystream-Generierung über eine native
    <code>
     CC_OpenCL.dll
    </code>
    nutzt. Es wurde entwickelt, um die Herausforderungen der Datenintegrität, Vertraulichkeit und des Wiederverwendungsschutzes in modernen Datenverarbeitungsumgebungen zu adressieren, insbesondere dort, wo hohe Durchsätze und starke kryptographische Garantien erforderlich sind. Durch die Integration von Authenticated Encryption with Associated Data (AEAD) mit BLAKE3-MAC und Nonce/Counter-Mechanismen bietet SQSE AEAD eine robuste Lösung gegen Manipulationen und Keystream-Wiederverwendung.
   </p>
   <p>
    Das System richtet sich an Entwickler, Sicherheitsexperten und Unternehmen, die eine fortschrittliche, GPU-gestützte Verschlüsselungslösung für sensible Daten benötigen. Es bietet flexible Schlüsselableitungsoptionen (Passwort mit Argon2id, Seed, NPY-Rohkeys) und benutzerfreundliche Schnittstellen (CLI, Streamlit GUI), die die Komplexität der zugrundeliegenden kryptographischen Operationen abstrahieren, während gleichzeitig ein hohes Maß an Kontrolle und Transparenz über die Sicherheitsparameter gewährleistet wird. Die Kombination aus neuartigen Keystream-Generatoren und etablierten kryptographischen Primitiven positioniert SQSE AEAD als eine praktische und leistungsfähige Alternative in der modernen Kryptographie.
   </p>
  </section>
  <section>
   <h2>
    3. Problemstellung
   </h2>
   <p>
    In der heutigen digitalen Landschaft sind Datenschutz und Datensicherheit von größter Bedeutung. Herkömmliche Verschlüsselungsmethoden stehen vor vielfältigen Herausforderungen, darunter:
   </p>
   <ul>
    <li>
     <strong>
      Performance-Engpässe:
     </strong>
     Die Verschlüsselung großer Datenmengen kann auf CPU-basierten Systemen zeitaufwändig sein, insbesondere bei Echtzeitanwendungen oder Big-Data-Szenarien.
    </li>
    <li>
     <strong>
      Sicherheitslücken durch Keystream-Wiederverwendung:
     </strong>
     Bei Stromchiffren kann die Wiederverwendung desselben Keystreams mit unterschiedlichen Plaintexts zu schwerwiegenden Sicherheitslücken führen, die eine Entschlüsselung ermöglichen.
    </li>
    <li>
     <strong>
      Fehlende Integritätsprüfung:
     </strong>
     Viele einfache Verschlüsselungsschemata bieten keine Mechanismen, um zu erkennen, ob Daten nach der Verschlüsselung manipuliert wurden, was zu unbemerkten Datenkorruptionen oder gezielten Angriffen führen kann.
    </li>
    <li>
     <strong>
      Schlüsselmanagement-Komplexität:
     </strong>
     Die sichere Erzeugung, Speicherung und Ableitung von Schlüsseln, insbesondere aus Passwörtern, erfordert robuste Key Derivation Functions (KDFs) und Mechanismen zum Schutz vor Brute-Force-Angriffen.
    </li>
    <li>
     <strong>
      Bedrohung durch zukünftige Angriffe:
     </strong>
     Obwohl SQSE keine formale Post-Quanten-Kryptographie darstellt, adressiert sie durch ihre chaotischen dynamischen Systeme und GPU-Beschleunigung die Notwendigkeit, über traditionelle Ansätze hinauszugehen und eine praktisch robuste Keystream-Quelle zu bieten.
    </li>
   </ul>
   <p>
    Das Projekt "SQSE AEAD" zielt darauf ab, diese Probleme durch die Kombination einer GPU-beschleunigten, chaotischen Keystream-Generierung mit einem modernen AEAD-Konstrukt zu lösen, das sowohl Vertraulichkeit als auch Authentizität gewährleistet.
   </p>
  </section>
  <section>
   <h2>
    4. Systemarchitektur und Funktionsweise
   </h2>
   <p>
    Die Architektur von SQSE AEAD ist modular aufgebaut und in mehrere Schichten unterteilt, um eine klare Trennung der Verantwortlichkeiten und eine hohe Wartbarkeit zu gewährleisten. Der Kern des Systems ist auf Effizienz und Sicherheit ausgelegt.
   </p>
   <h3>
    4.1. Architekturdiagramm
   </h3>
   <div class="mermaid">
    sequenceDiagram
    actor User
    box "Application Layer"
        participant CLI_GUI as "CLI / Streamlit GUI"
    end

    box "Python Core Logic (sqse_aead_pkg)"
        participant Core_AEAD_Logic as "Core AEAD Logic (core.py)"
        participant Key_Derivation as "Key Derivation (core.py)"
        participant SQSE_DLL_Wrapper as "SQSE DLL Wrapper (core.py)"
    end

    box "Native GPU Layer"
        participant CC_OpenCL_DLL as "CC_OpenCL.dll (GPU Keystream)"
    end

    box "External Libraries"
        participant External_Crypto_Libs as "Blake3 / Argon2"
        participant Filesystem as "Filesystem I/O"
    end

    User-&gt;&gt;CLI_GUI: Start Encrypt/Decrypt (file, key_source, params)

    CLI_GUI-&gt;&gt;Core_AEAD_Logic: call encrypt_file(infile, outfile, ...) / decrypt_file(infile, outfile, ...)
    activate Core_AEAD_Logic

    Core_AEAD_Logic-&gt;&gt;Filesystem: Read plaintext file size (Encrypt) / Read Header (Decrypt)
    Core_AEAD_Logic-&gt;&gt;Core_AEAD_Logic: Generate Salt &amp; Nonce (Encrypt)

    Core_AEAD_Logic-&gt;&gt;Key_Derivation: Determine Key Source (Password/Seed/NPY)
    activate Key_Derivation
    alt Password (Argon2id)
        Key_Derivation-&gt;&gt;External_Crypto_Libs: kdf_from_password(password, salt, ...)
        External_Crypto_Libs--&gt;&gt;Key_Derivation: Derived Master Key (32 bytes)
    else Seed (Blake3)
        Key_Derivation-&gt;&gt;External_Crypto_Libs: kdf_from_seed(seed)
        External_Crypto_Libs--&gt;&gt;Key_Derivation: Derived Master Key (32 bytes)
    else NPY (Blake3)
        Key_Derivation-&gt;&gt;External_Crypto_Libs: kdf_from_npy(key_npy_path)
        External_Crypto_Libs--&gt;&gt;Key_Derivation: Derived Master Key (32 bytes)
    end
    Key_Derivation--&gt;&gt;Core_AEAD_Logic: Master Key
    deactivate Key_Derivation

    Core_AEAD_Logic-&gt;&gt;SQSE_DLL_Wrapper: Initialize SQSEDLL(dll_path, gpu_idx)
    activate SQSE_DLL_Wrapper
    SQSE_DLL_Wrapper-&gt;&gt;CC_OpenCL_DLL: Load DLL &amp; Init GPU (initialize_gpu, sqse_load_kernels)
    activate CC_OpenCL_DLL
    CC_OpenCL_DLL--&gt;&gt;SQSE_DLL_Wrapper: DLL Ready
    deactivate CC_OpenCL_DLL
    SQSE_DLL_Wrapper--&gt;&gt;Core_AEAD_Logic: SQSEDLL Instance
    deactivate SQSE_DLL_Wrapper

    Core_AEAD_Logic-&gt;&gt;Filesystem: Write Header (MAGIC, VER, Flags, KDF params, K, Steps, Nonce, Salt) (Encrypt)
    Core_AEAD_Logic-&gt;&gt;External_Crypto_Libs: Init BLAKE3 MAC with Master Key
    activate External_Crypto_Libs
    Core_AEAD_Logic-&gt;&gt;External_Crypto_Libs: Update MAC with Header data

    loop For each data chunk
        Core_AEAD_Logic-&gt;&gt;Filesystem: Read Plaintext Chunk (Encrypt) / Ciphertext Chunk (Decrypt)
        
        Core_AEAD_Logic-&gt;&gt;External_Crypto_Libs: expand_key_float32(Master Key, Nonce, Chunk Counter)
        External_Crypto_Libs--&gt;&gt;Core_AEAD_Logic: Chunk-specific Float32 Key Array

        Core_AEAD_Logic-&gt;&gt;SQSE_DLL_Wrapper: keystream_bytes(chunk_size, K, Steps, Float32 Key Array)
        activate SQSE_DLL_Wrapper
        SQSE_DLL_Wrapper-&gt;&gt;CC_OpenCL_DLL: execute_sqse_encrypt_float(zeros_input, Float32 Key Array, ...)
        activate CC_OpenCL_DLL
        CC_OpenCL_DLL--&gt;&gt;SQSE_DLL_Wrapper: theta, pmask (Keystream components)
        deactivate CC_OpenCL_DLL
        SQSE_DLL_Wrapper-&gt;&gt;SQSE_DLL_Wrapper: Convert theta to byte Keystream
        SQSE_DLL_Wrapper--&gt;&gt;Core_AEAD_Logic: Byte Keystream
        deactivate SQSE_DLL_Wrapper

        Core_AEAD_Logic-&gt;&gt;Core_AEAD_Logic: XOR (Plaintext/Ciphertext) with Keystream
        Core_AEAD_Logic-&gt;&gt;Filesystem: Write Ciphertext Chunk (Encrypt) / Plaintext Chunk (Decrypt)
        Core_AEAD_Logic-&gt;&gt;External_Crypto_Libs: Update BLAKE3 MAC with Ciphertext Chunk
    end

    Core_AEAD_Logic-&gt;&gt;Filesystem: Write Final BLAKE3 MAC (Encrypt)
    Core_AEAD_Logic-&gt;&gt;Filesystem: Read Stored BLAKE3 MAC (Decrypt)
    Core_AEAD_Logic-&gt;&gt;External_Crypto_Libs: Finalize BLAKE3 MAC
    External_Crypto_Libs--&gt;&gt;Core_AEAD_Logic: Calculated MAC

    alt Decryption
        Core_AEAD_Logic-&gt;&gt;Core_AEAD_Logic: Compare Calculated MAC with Stored MAC
        alt MAC Mismatch
            Core_AEAD_Logic--xCLI_GUI: Error: MAC Verification Failed (Tampered/Wrong Key)
        else MAC Match
            Core_AEAD_Logic--&gt;&gt;CLI_GUI: Decryption Successful
        end
    else Encryption
        Core_AEAD_Logic--&gt;&gt;CLI_GUI: Encryption Successful
    end
    deactivate External_Crypto_Libs
    deactivate Core_AEAD_Logic
    CLI_GUI--&gt;&gt;User: Operation Result
   </div>
   <p class="figure-caption">
    Abbildung 1: Systemarchitektur und Datenfluss von SQSE AEAD
   </p>
   <h3>
    4.2. Komponentenbeschreibung
   </h3>
   <ul>
    <li>
     <strong>
      Application Layer (CLI / Streamlit GUI):
     </strong>
     Bietet die Benutzerschnittstelle für Interaktionen. Die CLI (
     <code>
      sqse_aead_pkg.cli
     </code>
     ) nutzt Typer für eine robuste Kommandozeilenbedienung, während die Streamlit GUI (
     <code>
      app_streamlit_aead.py
     </code>
     ) eine interaktive Weboberfläche für Dateiauswahl, Parameterkonfiguration und Live-Logging bereitstellt.
    </li>
    <li>
     <strong>
      Python Core Logic (
      <code>
       sqse_aead_pkg.core
      </code>
      ):
     </strong>
     Das Herzstück der Anwendung. Es koordiniert den gesamten Verschlüsselungs- und Entschlüsselungsprozess. Dazu gehören:
     <ul>
      <li>
       <strong>
        Schlüsselableitung:
       </strong>
       Verwaltet die Ableitung des Master-Schlüssels aus Passwörtern (mittels Argon2id für Memory-Hardness), Seeds oder NPY-Rohkeys (mittels BLAKE3).
      </li>
      <li>
       <strong>
        Dateiverarbeitung:
       </strong>
       Liest und schreibt Dateien in Chunks, um große Datenmengen effizient zu verarbeiten.
      </li>
      <li>
       <strong>
        Header-Management:
       </strong>
       Erstellt und parst den Dateikopf (Header-v2), der Metadaten wie Version, Flags, KDF-Parameter (Argon2), SQSE-Parameter (K, Steps), Chunk-Größe, Gesamtgröße, Salt und Nonce enthält.
      </li>
      <li>
       <strong>
        AEAD-Koordination:
       </strong>
       Stellt die Authenticated Encryption with Associated Data (AEAD) Funktionalität sicher, indem es Nonces und Counter für den Wiederverwendungsschutz verwendet und den BLAKE3-MAC für Integrität und Authentizität des Headers und Ciphertexts berechnet und verifiziert.
      </li>
     </ul>
    </li>
    <li>
     <strong>
      SQSE DLL Wrapper (
      <code>
       sqse_aead_pkg.core
      </code>
      ):
     </strong>
     Eine Python-Schnittstelle (mittels
     <code>
      ctypes
     </code>
     ) zur nativen
     <code>
      CC_OpenCL.dll
     </code>
     . Dieser Wrapper lädt die DLL, initialisiert die GPU und ruft die Low-Level-Funktionen zur Keystream-Generierung auf.
    </li>
    <li>
     <strong>
      Native GPU Layer (
      <code>
       CC_OpenCL.dll
      </code>
      ):
     </strong>
     Die GPU-beschleunigte Bibliothek, die die eigentliche "Sub-Quantum State Encryption" (SQSE) Engine implementiert. Diese DLL nutzt OpenCL, um chaotische dynamische Systeme (z.B. Chirikov-Map-ähnliche Iterationen) auf der GPU auszuführen. Aus einem Master-Schlüssel und einem Chunk-spezifischen Kontextschlüssel wird ein deterministischer Keystream generiert.
    </li>
    <li>
     <strong>
      External Libraries (BLAKE3 / Argon2):
     </strong>
     <ul>
      <li>
       <strong>
        BLAKE3:
       </strong>
       Wird als Hash-Funktion für die Schlüsselableitung aus Seeds/NPY-Dateien, für die Expansion chunk-spezifischer Schlüssel und als Message Authentication Code (MAC) für die Integritätsprüfung verwendet.
      </li>
      <li>
       <strong>
        Argon2id:
       </strong>
       Eine Memory-Hard Key Derivation Function (KDF), die für die sichere Ableitung von Schlüsseln aus Passwörtern eingesetzt wird. Sie schützt effektiv vor Brute-Force- und Wörterbuchangriffen durch hohen Speicher- und Zeitaufwand.
      </li>
     </ul>
    </li>
    <li>
     <strong>
      Filesystem I/O:
     </strong>
     Standardmäßige Dateisystemoperationen zum Lesen von Plaintext- und Schreiben von Ciphertext-Dateien sowie umgekehrt.
    </li>
   </ul>
   <h3>
    4.3. Funktionsweise
   </h3>
   <p>
    Der Verschlüsselungsprozess beginnt mit der Auswahl einer Eingabedatei und der Konfiguration der Schlüsselquelle (Passwort, Seed oder NPY-Datei) sowie der SQSE-Parameter (Chaos-Parameter K, Iterationsschritte). Die Kernlogik leitet den Master-Schlüssel ab und generiert ein zufälliges Salt und eine Nonce. Anschließend wird ein Header erstellt, der alle notwendigen Metadaten für die spätere Entschlüsselung enthält, einschließlich der Argon2id-Parameter, falls ein Passwort verwendet wird. Dieser Header wird zusammen mit dem Salt und der Nonce in die Ausgabedatei geschrieben.
   </p>
   <p>
    Die Eingabedatei wird in Chunks fester Größe (standardmäßig 8 MiB) gelesen. Für jeden Chunk wird ein eindeutiger Kontextschlüssel mithilfe des Master-Schlüssels, der Nonce und eines inkrementellen Counters abgeleitet. Dieser Kontextschlüssel wird dann an die
    <code>
     CC_OpenCL.dll
    </code>
    übergeben, die auf der GPU einen deterministischen Keystream generiert. Der generierte Keystream wird bitweise mit dem Daten-Chunk XOR-verknüpft, um den Ciphertext-Chunk zu erzeugen. Jeder Ciphertext-Chunk wird direkt in die Ausgabedatei geschrieben und gleichzeitig in die BLAKE3-MAC-Berechnung einbezogen. Nach Verarbeitung aller Chunks wird der finale BLAKE3-MAC an den Ciphertext angehängt.
   </p>
   <p>
    Die Entschlüsselung folgt dem umgekehrten Prinzip. Zuerst wird der Header gelesen, um die ursprünglichen Parameter zu rekonstruieren. Basierend auf der Schlüsselquelle wird der Master-Schlüssel erneut abgeleitet. Der Ciphertext wird Chunk für Chunk gelesen, der Keystream für jeden Chunk generiert und mit dem Ciphertext XOR-verknüpft, um den Plaintext zu rekonstruieren. Gleichzeitig wird der BLAKE3-MAC über den gelesenen Ciphertext und Header berechnet und mit dem im Dateiende gespeicherten MAC verglichen. Bei einer Diskrepanz wird ein Manipulationsalarm ausgelöst, was die Integrität und Authentizität der Daten garantiert.
   </p>
  </section>
  <section>
   <h2>
    5. Evaluation und Testergebnisse
   </h2>
   <p>
    Die Evaluation von SQSE AEAD konzentriert sich auf Robustheit, Performance und Benutzerfreundlichkeit, gestützt durch interne Tests und Kostenabschätzungen.
   </p>
   <h3>
    5.1. Robustheit
   </h3>
   <p>
    Die Robustheit des Systems wird durch mehrere Schichten gewährleistet:
   </p>
   <ul>
    <li>
     <strong>
      Kryptographische Primitiven:
     </strong>
     Einsatz von BLAKE3 für MAC und Schlüsselableitung sowie Argon2id für Passwort-Hashing bietet etablierte Sicherheitsstandards.
    </li>
    <li>
     <strong>
      AEAD-Konstrukt:
     </strong>
     Die Verwendung von Nonce/Counter pro Chunk verhindert Keystream-Wiederverwendung, und der BLAKE3-MAC schützt vor unbemerkten Manipulationen.
    </li>
    <li>
     <strong>
      DLL-Integrität:
     </strong>
     Die Notwendigkeit, die
     <code>
      CC_OpenCL.dll
     </code>
     zu schützen (z.B. durch Signaturen und Integritäts-Checks), wird explizit hervorgehoben und durch CI-Workflows unterstützt, die einen Mock-Stub für Linux und den Download einer Test-DLL für Windows-Runner vorsehen.
    </li>
    <li>
     <strong>
      Selftest-Mechanismus:
     </strong>
     Der integrierte Selftest (
     <code>
      selftest_cmd
     </code>
     ) validiert den vollständigen Encrypt-Decrypt-Roundtrip und die MAC-Verifikation, um die korrekte Funktion des Systems zu bestätigen.
    </li>
   </ul>
   <h3>
    5.2. Performance / Geschwindigkeit
   </h3>
   <p>
    Die GPU-Beschleunigung der Keystream-Generierung durch die
    <code>
     CC_OpenCL.dll
    </code>
    ist ein zentrales Leistungsmerkmal. Obwohl keine konkreten MB/s-Benchmarks direkt im Code enthalten sind, impliziert die Nutzung von OpenCL und GPUs eine signifikante Beschleunigung gegenüber reinen CPU-Implementierungen, insbesondere bei großen Datenmengen. Die Chunk-basierte Verarbeitung ermöglicht eine effiziente Nutzung der GPU-Ressourcen und eine skalierbare Verarbeitung von Dateien beliebiger Größe.
   </p>
   <h3>
    5.3. Benutzerfreundlichkeit (UX)
   </h3>
   <ul>
    <li>
     <strong>
      CLI:
     </strong>
     Die Typer-basierte Kommandozeilenschnittstelle ist intuitiv und bietet klare Optionen für Verschlüsselung, Entschlüsselung und Selbstdurchführung.
    </li>
    <li>
     <strong>
      Streamlit GUI:
     </strong>
     Bietet eine benutzerfreundliche grafische Oberfläche mit Dateiauswahl, Key-Optionen, GPU-Index-Auswahl und Live-Logs, was die Zugänglichkeit für Nicht-CLI-Benutzer erheblich verbessert.
    </li>
    <li>
     <strong>
      Flexible Schlüsseloptionen:
     </strong>
     Unterstützung für Passwörter (mit Argon2id), deterministische Seeds und NPY-Rohkeys deckt verschiedene Anwendungsfälle ab.
    </li>
   </ul>
   <h3>
    5.4. Kosten- und Angriffsabschätzung
   </h3>
   <p>
    Die Dokumentation (
    <code>
     Info_extended.md
    </code>
    ,
    <code>
     cost_estimate.md
    </code>
    ) enthält detaillierte Kostenabschätzungen für Angreifer, die die Wirksamkeit von Argon2id und starken Schlüsseln unterstreichen:
   </p>
   <table>
    <thead>
     <tr>
      <th>
       Entropie (Bits)
      </th>
      <th>
       Profile
      </th>
      <th>
       Erwartete Jahre
      </th>
      <th>
       Benötigte GPU‑Hours (1 GPU)
      </th>
      <th>
       Geschätzte Kosten (1 GPU)
      </th>
     </tr>
    </thead>
    <tbody>
     <tr>
      <td>
       40
      </td>
      <td>
       Low-cost GPU (R=1e9/s, $3/GPUh)
      </td>
      <td>
       0.00
      </td>
      <td>
       0
      </td>
      <td>
       $0.46
      </td>
     </tr>
     <tr>
      <td>
       40
      </td>
      <td>
       Moderate (R=1e8/s, $1/GPUh)
      </td>
      <td>
       0.00
      </td>
      <td>
       2
      </td>
      <td>
       $1.53
      </td>
     </tr>
     <tr>
      <td>
       40
      </td>
      <td>
       Argon2-heavy (R=2/s per worker, $0.05/worker-hour)
      </td>
      <td>
       8710.36
      </td>
      <td>
       7.635e+07
      </td>
      <td>
       $3.818e+06
      </td>
     </tr>
     <tr>
      <td>
       56
      </td>
      <td>
       Low-cost GPU (R=1e9/s, $3/GPUh)
      </td>
      <td>
       1.14
      </td>
      <td>
       10,008
      </td>
      <td>
       $30,024.00
      </td>
     </tr>
     <tr>
      <td>
       56
      </td>
      <td>
       Moderate (R=1e8/s, $1/GPUh)
      </td>
      <td>
       11.42
      </td>
      <td>
       100,080
      </td>
      <td>
       $100,079.99
      </td>
     </tr>
     <tr>
      <td>
       56
      </td>
      <td>
       Argon2-heavy (R=2/s per worker, $0.05/worker-hour)
      </td>
      <td>
       5.708e+08
      </td>
      <td>
       5.004e+12
      </td>
      <td>
       $2.502e+11
      </td>
     </tr>
     <tr>
      <td>
       64
      </td>
      <td>
       Low-cost GPU (R=1e9/s, $3/GPUh)
      </td>
      <td>
       292.27
      </td>
      <td>
       2.562e+06
      </td>
      <td>
       $7.686e+06
      </td>
     </tr>
     <tr>
      <td>
       64
      </td>
      <td>
       Moderate (R=1e8/s, $1/GPUh)
      </td>
      <td>
       2922.71
      </td>
      <td>
       2.562e+07
      </td>
      <td>
       $2.562e+07
      </td>
     </tr>
     <tr>
      <td>
       64
      </td>
      <td>
       Argon2-heavy (R=2/s per worker, $0.05/worker-hour)
      </td>
      <td>
       1.461e+11
      </td>
      <td>
       1.281e+15
      </td>
      <td>
       $6.405e+13
      </td>
     </tr>
     <tr>
      <td>
       80
      </td>
      <td>
       Low-cost GPU (R=1e9/s, $3/GPUh)
      </td>
      <td>
       1.915e+07
      </td>
      <td>
       1.679e+11
      </td>
      <td>
       $5.037e+11
      </td>
     </tr>
     <tr>
      <td>
       80
      </td>
      <td>
       Moderate (R=1e8/s, $1/GPUh)
      </td>
      <td>
       1.915e+08
      </td>
      <td>
       1.679e+12
      </td>
      <td>
       $1.679e+12
      </td>
     </tr>
     <tr>
      <td>
       80
      </td>
      <td>
       Argon2-heavy (R=2/s per worker, $0.05/worker-hour)
      </td>
      <td>
       9.577e+15
      </td>
      <td>
       8.395e+19
      </td>
      <td>
       $4.198e+18
      </td>
     </tr>
    </tbody>
   </table>
   <p>
    Diese Zahlen verdeutlichen, dass Angriffe auf nicht-Memory-Hard-Ziele mit geringer Entropie (≤64 Bit) potenziell kostengünstig sind, während Argon2-geschützte Passwörter die Angriffsrate drastisch reduzieren und Brute-Force-Angriffe ökonomisch unrentabel machen. Schlüssel mit 128 Bit und mehr sind praktisch unknackbar.
   </p>
  </section>
  <section>
   <h2>
    6. Vergleich mit anderen Tools
   </h2>
   <p>
    SQSE AEAD unterscheidet sich von traditionellen Verschlüsselungstools und -bibliotheken durch seine einzigartige Kombination aus GPU-beschleunigter Keystream-Generierung und einem robusten AEAD-Konstrukt. Im Vergleich zu:
   </p>
   <ul>
    <li>
     <strong>
      Standard-Stromchiffren (z.B. AES-CTR, XChaCha20-Poly1305):
     </strong>
     Während diese etablierte Algorithmen verwenden, bietet SQSE eine alternative Keystream-Quelle, die auf chaotischen dynamischen Systemen basiert. Der Vorteil liegt in der potenziellen Performance-Skalierung durch GPU-Nutzung, insbesondere bei sehr hohen Datenraten, und der Forschungsperspektive, die über rein algebraische Konstruktionen hinausgeht.
    </li>
    <li>
     <strong>
      Dateiverschlüsselungstools (z.B. GnuPG, VeraCrypt):
     </strong>
     Diese bieten umfassende Dateisystem- oder Container-Verschlüsselung. SQSE AEAD konzentriert sich auf die effiziente und sichere Verschlüsselung einzelner Dateien oder Datenströme mit starkem Fokus auf AEAD und GPU-Beschleunigung. Es ist eher als eine Komponente für größere Systeme oder als spezialisiertes Tool für Hochleistungsanwendungen gedacht.
    </li>
   </ul>
   <h3>
    6.1. Alleinstellungsmerkmale und Differenzierungsfaktoren
   </h3>
   <ul>
    <li>
     <strong>
      GPU-beschleunigte Keystream-Generierung:
     </strong>
     Die native
     <code>
      CC_OpenCL.dll
     </code>
     nutzt die Parallelisierungsfähigkeiten moderner GPUs, um Keystreams mit hoher Geschwindigkeit zu erzeugen. Dies ist ein entscheidender Vorteil für Anwendungen, die einen hohen Datendurchsatz erfordern.
    </li>
    <li>
     <strong>
      Sub-Quantum State Encryption (SQSE):
     </strong>
     Das Projekt basiert auf einem neuartigen Ansatz, der chaotische dynamische Systeme für die Keystream-Generierung verwendet. Obwohl es keine formale Post-Quanten-Kryptographie ist, bietet es eine praktisch robuste Keystream-Quelle, die hochgradig sensitiv gegenüber Schlüssel- und Parameteränderungen ist.
    </li>
    <li>
     <strong>
      Integrierte AEAD-Funktionalität:
     </strong>
     Die konsequente Anwendung von Nonce/Counter und BLAKE3-MAC gewährleistet sowohl Vertraulichkeit als auch Authentizität und Integrität, was ein grundlegendes Merkmal moderner kryptographischer Systeme ist.
    </li>
    <li>
     <strong>
      Flexible Schlüsselableitung:
     </strong>
     Die Unterstützung von Argon2id für Passwörter, BLAKE3 für Seeds und NPY-Rohkeys bietet eine hohe Flexibilität und Anpassungsfähigkeit an unterschiedliche Sicherheitsanforderungen und Schlüsselmanagement-Strategien, einschließlich HSM-Integration.
    </li>
    <li>
     <strong>
      Modularität:
     </strong>
     Die Trennung der Keystream-Generierung von der AEAD-Konstruktion ermöglicht eine modulare Austauschbarkeit der Keystream-Quelle, was zukünftige Erweiterungen und Anpassungen erleichtert.
    </li>
   </ul>
  </section>
  <section>
   <h2>
    7. Kernkonzepte und Innovationen
   </h2>
   <p>
    SQSE AEAD integriert mehrere Kernkonzepte und innovative Ansätze, die es von herkömmlichen Verschlüsselungssystemen abheben:
   </p>
   <h3>
    7.1. Sub-Quantum State Encryption (SQSE)
   </h3>
   <p>
    Die zentrale Innovation ist die Nutzung von SQSE als Keystream-Quelle. Anstatt auf traditionellen algebraischen Operationen zu basieren, verwendet SQSE gekoppelte dynamische Systeme, die chaotische Karten (z.B. Chirikov-/Standard-Map-artige Iterationen) und feldbasierte Iterationen mit internen Rückkopplungen umfassen. Die Sicherheit der SQSE-Engine beruht auf der hohen Sensitivität gegenüber initialen Bedingungen (Schlüssel, Seed, Parameter), was eine praktische Nicht-Invertierbarkeit ohne das Geheimnis zur Folge hat. Die GPU-Beschleunigung durch OpenCL-Kernel ermöglicht die effiziente Berechnung dieser komplexen Dynamiken.
   </p>
   <h3>
    7.2. Authenticated Encryption with Associated Data (AEAD)
   </h3>
   <p>
    SQSE AEAD ist nicht nur eine Keystream-Quelle, sondern ein vollständiges AEAD-Konstrukt. Dies bedeutet, dass es nicht nur die Vertraulichkeit (Verschlüsselung) der Daten gewährleistet, sondern auch deren Integrität und Authentizität. Dies wird erreicht durch:
   </p>
   <ul>
    <li>
     <strong>
      Nonce/Counter-Mechanismus:
     </strong>
     Für jeden Daten-Chunk wird ein eindeutiger Kontextschlüssel unter Verwendung einer Nonce und eines inkrementellen Counters abgeleitet. Dies verhindert die fatale Keystream-Wiederverwendung, die bei reinen XOR-Chiffren ein großes Risiko darstellt.
    </li>
    <li>
     <strong>
      BLAKE3-MAC:
     </strong>
     Ein hochperformanter und sicherer Message Authentication Code (MAC) wird über den gesamten Header und alle Ciphertext-Chunks berechnet. Dies ermöglicht es dem Empfänger, jede Manipulation der Daten oder des Headers zu erkennen und die Authentizität des Absenders (im Kontext des gemeinsamen Schlüssels) zu verifizieren.
    </li>
   </ul>
   <h3>
    7.3. Memory-Hard Key Derivation (Argon2id)
   </h3>
   <p>
    Für die Schlüsselableitung aus Passwörtern wird Argon2id verwendet, eine preisgekrönte Memory-Hard Key Derivation Function. Ihre Parameter (Speicherverbrauch
    <code>
     m
    </code>
    , Zeitkosten
    <code>
     t
    </code>
    , Parallelität
    <code>
     p
    </code>
    ) werden im Header der verschlüsselten Datei gespeichert und bei der Entschlüsselung wiederverwendet. Dies macht Brute-Force-Angriffe auf Passwörter extrem teuer und zeitaufwändig, da sie einen hohen Speicher- und Rechenaufwand pro Versuch erfordern.
   </p>
   <h3>
    7.4. Modulare Schlüsselquellen
   </h3>
   <p>
    Das System unterstützt flexible Schlüsselquellen:
   </p>
   <ul>
    <li>
     <strong>
     </strong>
    </li>
   </ul>
  </section>
 </div>
</body>
