package enigma;

import java.util.HashMap;

/** Represents a permutation of a range of integers starting at 0 corresponding
 *  to the characters of an alphabet.
 *  @author danyal
 */
class Permutation {

    /** Set this Permutation to that specified by CYCLES, a string in the
     *  form "(cccc) (cc) ..." where the c's are characters in ALPHABET, which
     *  is interpreted as a permutation in cycle notation.  Characters in the
     *  alphabet that are not included in any cycle map to themselves.
     *  Whitespace is ignored. */
    private HashMap<Integer, Character>
            cycleList = new HashMap<Integer, Character>();

    /** Sets up a permutation.
     * @param alphabet it is an alphabet
     * @param cycles it is a cycle */
    Permutation(String cycles, Alphabet alphabet) {
        _alphabet = alphabet;
        _cycles = cycles;
        cycles = cycles.replaceAll(" ", "");
        char[] c = cycles.toCharArray();
        addCycle(c);

        int left = 0;
        int right = 0;
        for (int i = 0; i < cycleList.size(); i++) {
            char curr = cycleList.get(i);
            if (curr == '(') {
                left++;
            }
            if (curr == ')') {
                right++;
            }
        }
        if (left != right) {
            throw new EnigmaException("Invalid input.");
        }
    }

    /** Add the cycle c0->c1->...->cm->c0 to the permutation, where CYCLE is
     *  c0c1...cm. */
    private void addCycle(char[] cycle) {
        int index = 0;
        for (char i : cycle) {
            cycleList.put(index, i);
            index++;
        }
    }

    /** Return the value of P modulo the size of this permutation. */
    final int wrap(int p) {
        int r = p % size();
        if (r < 0) {
            r += size();
        }
        return r;
    }

    /** Returns the size of the alphabet I permute. */
    int size() {
        return alphabet().size();
    }

    /** Return the result of applying this permutation to P modulo the
     *  alphabet size. */
    int permute(int p) {
        int perm = wrap(p);
        char toAlph = alphabet().toChar(perm);
        if (cycleList.containsValue(toAlph)) {
            char store = 0;
            for (int i = 0; i < cycleList.size(); i++) {
                if (cycleList.get(i) == "(".toCharArray()[0]) {
                    store = cycleList.get(i + 1);
                }
                if (cycleList.get(i) == toAlph) {
                    if (_alphabet.contains(cycleList.get(i + 1))) {
                        return alphabet().toInt(cycleList.get(i + 1));
                    }
                    return _alphabet.toInt(store);
                }
            }

        }
        return perm;
    }

    /** Return the result of applying the inverse of this permutation
     *  to  C modulo the alphabet size. */
    int invert(int c) {
        int inverse = wrap(c);
        char toAlph = alphabet().toChar(inverse);
        if (cycleList.containsValue(toAlph)) {
            char store = 0;
            boolean check = false;
            for (int i = 0; i < cycleList.size(); i++) {
                if (cycleList.get(i) == toAlph) {
                    check = true;
                    if (_alphabet.contains(cycleList.get(i - 1))) {
                        return _alphabet.toInt(cycleList.get(i - 1));
                    }
                }
                if (cycleList.get(i) == ")".toCharArray()[0]) {
                    store = cycleList.get(i - 1);
                    if (check) {
                        return _alphabet.toInt(store);
                    }
                }
            }
        }
        return inverse;
    }

    /** Return the result of applying this permutation to the index of P
     *  in ALPHABET, and converting the result to a character of ALPHABET. */
    char permute(char p) {
        char store = 0;
        for (int i = 0; i < cycleList.size(); i++) {
            if (cycleList.get(i) == "(".toCharArray()[0]) {
                store = cycleList.get(i + 1);
            }
            if (cycleList.get(i) == p
                    && _alphabet.contains(cycleList.get(i + 1))) {
                return cycleList.get(i + 1);
            } else if (cycleList.get(i) == p) {
                return store;
            }
        }
        return p;
    }
    /** Return the result of applying the inverse of this permutation to C. */
    char invert(char c) {
        if (cycleList.containsValue(c)) {
            char store = c;
            boolean check = false;
            for (int i = 0; i < cycleList.size(); i++) {
                if (cycleList.get(i) == c) {
                    check = true;
                    if (_alphabet.contains(cycleList.get(i - 1))) {
                        return cycleList.get(i - 1);
                    }
                }
                if (cycleList.get(i) == ")".toCharArray()[0]) {
                    store = cycleList.get(i - 1);
                    if (check) {
                        return store;
                    }
                }
            }
        }
        return c;
    }

    /** Return the alphabet used to initialize this Permutation. */
    Alphabet alphabet() {
        return _alphabet;
    }

    /** Return true iff this permutation is a derangement (i.e., a
     *  permutation for which no value maps to itself). */
    boolean derangement() {
        for (int i = 0; i < cycleList.size(); i++) {
            if (permute(cycleList.get(i)) == cycleList.get(i)) {
                return false;
            }
        }
        return true;
    }

    /** Alphabet of this permutation. */
    private Alphabet _alphabet;

    /** The cycles which are inputed into a permutation. */
    private String _cycles;
}



