/***************************************************************************
*     2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/
#pragma once

#include "../device_vector.h"

#include "bolt/amp/iterator/iterator_traits.h"
#include <cstddef>
#include <iterator>

namespace bolt
{
    namespace amp
    {
        template<typename T>
        class Ubiquitous_iterator : public std::iterator<device_vector_tag, T> {

            friend
            inline
            Ubiquitous_iterator operator+(
                Ubiquitous_iterator x, std::ptrdiff_t dx) [[cpu]][[hc]]
            {
                Ubiquitous_iterator y(x);
                return y += dx;
            }

            friend
            inline
            std::ptrdiff_t operator-(
                const Ubiquitous_iterator& x,
                const Ubiquitous_iterator& y) [[cpu]][[hc]]
            {
                return x.p_ - y.p_;
            }

            friend
            inline
            bool operator==(const Ubiquitous_iterator& x, const Ubiquitous_iterator& y) [[cpu]][[hc]]
            {
                return (x.p_ == y.p_);
            }

            friend
            inline
            bool operator!=(const Ubiquitous_iterator& x, const Ubiquitous_iterator& y) [[cpu]][[hc]]
            {
                return (x.p_ != y.p_);
            }



            T* p_;
        public:
            
            typedef typename std::iterator<device_vector_tag, T>::difference_type difference_type;
            // Public member variable 
            difference_type m_Index;
            Ubiquitous_iterator() [[cpu]][[hc]] = default;
            explicit
            Ubiquitous_iterator(T* p) [[cpu]][[hc]] : p_{p}, m_Index{0}{}

            T& operator[](std::ptrdiff_t dx) const [[cpu]][[hc]]
            { 
                return p_[dx];
            }
            T& operator[](std::ptrdiff_t dx) [[cpu]][[hc]] { return p_[dx]; }

            T& operator*() const [[cpu]][[hc]] { return *p_; }
            T& operator*() [[cpu]][[hc]] { return *p_; }

            Ubiquitous_iterator& operator+=(std::ptrdiff_t dx) [[cpu]][[hc]]
            {
                advance(dx);
                return *this;
            }

            Ubiquitous_iterator& operator++() [[cpu]][[hc]]
            {
                advance(1);
                return *this;
            }
            void advance(difference_type n) {
               m_Index += n;
               p_ += m_Index;
            }
            difference_type getIndex() const
            {
               return m_Index;
            }

            // Bolt glue.
            const Ubiquitous_iterator& getContainer() const { return *this; }
            Ubiquitous_iterator& getContainer() { return *this; }

            T* data() const { return p_; }
            T* data() { return p_; }

            concurrency::array_view<T> getBuffer(
                const Ubiquitous_iterator&, int sz) const
            {
                return concurrency::array_view<T>{sz, p_};
            }
            concurrency::array_view<T> getBuffer(
                const Ubiquitous_iterator&, int sz)
            {
                return concurrency::array<T>{sz, p_};
            }
            // Bolt glue.
        };
        template<typename T>
        inline
        Ubiquitous_iterator<T> make_ubiquitous_iterator(T* p) [[cpu]][[hc]]
        {
            return Ubiquitous_iterator<T>{p};
        }
    }
}
