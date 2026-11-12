#!/bin/bash
# =============================================================================
# run_production.sh  <seed> <job_index>
#
# job 하나 = seed 하나:
#   ttbar_nlo 복사 → run_card(nevents, iseed) → MG5 NLO launch (+ shower a)
#   → shower b, shower bkg (같은 LHE) → Delphes ×3 → 검증 → tar 로 반환
#
# 출력: production_results.tar.gz 안의 outputs/
#   sig_a_job<i>_seed<s>.root, sig_b_job<i>_seed<s>.root, bkg_job<i>_seed<s>.root
#   + 로그, 쓴 카드 사본, run_status.txt
# =============================================================================
set -eo pipefail                 # 어떤 명령이든 실패하면 즉시 종료 (pipe 안의 실패도 포함)

seed="$1"
job_index="$2"
NEVENTS=5000

# ---- 인자 검사: 숫자인지, seed 가 job_index 에서 나온 값과 맞는지 (실수 방지)
[[ "$job_index" =~ ^[0-9]+$ ]] || { echo 'job_index must be an integer' >&2; exit 10; }
expected_seed=$((3000001 + job_index))
[ "$seed" -eq "$expected_seed" ] || { echo "seed $seed != expected $expected_seed" >&2; exit 10; }

# ---- 경로 (여기만 환경에 맞게 고치면 됨)
batch=/home/syim/vts/ttbar_nlo        # 이 스크립트와 input_cards/ 가 있는 곳
master=/home/syim/vts/ttbar_nlo                       # compile 된 NLO process 디렉토리
mg5=/home/syim/hep/MG5_aMC_v3_8_0
input_cards="$batch/Cards"
PY8LIB=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/pythia8/312-1b262/x86_64-el9-gcc13-opt/lib
HEPMCLIB=/cvmfs/sft.cern.ch/lcg/releases/LCG_106/HepMC/2.06.11-XXXX/x86_64-el9-gcc13-opt/lib   # ← 실제 버전 경로로

job_scratch="$PWD"               # condor 가 job 마다 주는 작업 디렉토리
mkdir -p outputs

# ---- 종료 시 항상 실행: 상태 기록 + tar. (성공이든 실패든 로그는 돌려받기 위해)
finish() {
    job_rc=$?
    trap - EXIT
    printf 'JobIndex=%s\nSeed=%s\nEventsRequested=%s\nExitCode=%s\n' \
        "$job_index" "$seed" "$NEVENTS" "$job_rc" > "$job_scratch/outputs/run_status.txt"
    tar -czf "$job_scratch/production_results.tar.gz" -C "$job_scratch" outputs || exit 90
    exit "$job_rc"
}
trap finish EXIT

hostname; pwd

# ---- MG5 단계 환경 (대화형에서 성공한 조건 그대로: LCG view 는 아직 source 하지 않음)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$PY8LIB:$HEPMCLIB"
export OMP_NUM_THREADS=1

# ---- process 디렉토리 복사 (compile 된 상태 그대로)
test -r "$master/Cards/run_card.dat"
cp -a "$master" "$job_scratch/process"
process="$job_scratch/process"
rm -rf "$process/Events"/run_*                       # master 의 테스트 run 은 버림
cp -a "$input_cards" "$job_scratch/outputs/input_cards"

# ---- job 안에서는 MG5 가 condor 에 다시 제출하지 않도록 단일 코어로
cat >> "$process/Cards/amcatnlo_configuration.txt" <<EOF

run_mode = 0
nb_core = 1
automatic_html_opening = False
EOF

# ---- run_card: nevents, iseed 치환 (MG5 형식:  값 = 이름 ! 주석)
cp "$input_cards/run_card.dat" "$process/Cards/run_card.dat"
python3 - "$process/Cards/run_card.dat" "$NEVENTS" "$seed" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1]); text = path.read_text()
for key, val in (("nevents", sys.argv[2]), ("iseed", sys.argv[3])):
    pattern = rf"(?m)^\s*[^!#=\n]+\s*=\s*{key}\b[^\n]*$"
    text, n = re.subn(pattern, f"  {val} = {key} ! production job", text)
    assert n == 1, (key, n)
path.write_text(text)
PY

# ---- shower_card 준비: 변종별로 rnd_seed 만 job 마다 다르게 (형식:  rnd_seed = 0 # 주석)
set_shower_card() {              # $1 = 변종 이름 (a | b | bkg), $2 = seed 값
    python3 - "$input_cards/shower_card_$1.dat" "$process/Cards/shower_card.dat" "$2" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
text, n = re.subn(r"(?m)^\s*rnd_seed\s*=\s*\S+", f"rnd_seed     = {sys.argv[3]}", text)
assert n == 1, n
Path(sys.argv[2]).write_text(text)
PY
    cp "$process/Cards/shower_card.dat" "$job_scratch/outputs/input_cards/shower_card_$1.used.dat"
}

# ---- 방금 끝난 shower 의 HepMC 를 outputs/<변종>.hepmc.gz 로 옮김
#      (MG5 는 events_PYTHIA8_<n>.hepmc.gz 로 번호를 올려가며 저장하므로, 매번 남아 있는 걸 전부 옮긴다)
collect_hepmc() {                # $1 = 변종 이름
    local files=("$process/Events/run_prod/"events_PYTHIA8_*.hepmc.gz)
    [ "${#files[@]}" -eq 1 ] || { echo "expected exactly 1 hepmc for $1, got ${#files[@]}" >&2; exit 21; }
    mv "${files[0]}" "$job_scratch/outputs/$1.hepmc.gz"
}

# ============================================================== (1) NLO launch + shower a
set_shower_card a "$seed"
cat > "$job_scratch/outputs/cmd_launch.dat" <<EOF
launch $process -n run_prod
order=NLO
fixed_order=OFF
shower=PYTHIA8
madspin=OFF
reweight=OFF
done
done
EOF
python3 "$mg5/bin/mg5_aMC" "$job_scratch/outputs/cmd_launch.dat" > "$job_scratch/outputs/mg5_launch.log" 2>&1
cp "$process/Events/run_prod/"*.lhe.gz "$job_scratch/outputs/events.lhe.gz" 2>/dev/null || true
collect_hepmc sig_a

# ============================================================== (2) shower b, bkg  (같은 LHE)
cat > "$job_scratch/outputs/cmd_shower.dat" <<EOF
shower run_prod
done
EOF
set_shower_card b "$((seed + 100000))"
python3 "$process/bin/aMCatNLO" "$job_scratch/outputs/cmd_shower.dat" > "$job_scratch/outputs/mg5_shower_b.log" 2>&1
collect_hepmc sig_b

set_shower_card bkg "$((seed + 200000))"
python3 "$process/bin/aMCatNLO" "$job_scratch/outputs/cmd_shower.dat" > "$job_scratch/outputs/mg5_shower_bkg.log" 2>&1
collect_hepmc bkg

# ============================================================== (3) Delphes ×3  (여기서부터 LCG view 필요)
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
cp "$input_cards/delphes_card.tcl" "$input_cards/trackResolutionCMS.tcl" "$job_scratch/"   # source 되는 파일은 cwd 에
for v in sig_a sig_b bkg; do
    out="$job_scratch/outputs/${v}_job${job_index}_seed${seed}.root"
    gzip -dc "$job_scratch/outputs/$v.hepmc.gz" | DelphesHepMC2 "$job_scratch/delphes_card.tcl" "$out" \
        > "$job_scratch/outputs/delphes_$v.log" 2>&1
    rm "$job_scratch/outputs/$v.hepmc.gz"                # 크기 절약: ROOT 만 반환
done

# ============================================================== (4) 검증
python3 - "$job_scratch/outputs" "$job_index" "$seed" "$NEVENTS" > "$job_scratch/outputs/root_validation.log" 2>&1 <<'PY'
import sys, uproot
outdir, i, s, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for v in ("sig_a", "sig_b", "bkg"):
    with uproot.open(f"{outdir}/{v}_job{i}_seed{s}.root") as f:
        t = f["Delphes"]
        assert t.num_entries == n, (v, t.num_entries)
        for b in ("EFlowTrack/EFlowTrack.D0", "EFlowTrack/EFlowTrack.ErrorD0",
                  "Particle/Particle.M1", "Particle/Particle.X", "Jet/Jet.BTag"):
            t[b]
        print(v, "ok", t.num_entries)
PY
cat "$job_scratch/outputs/root_validation.log"
echo "job $job_index passed"
